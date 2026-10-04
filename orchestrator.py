import time
import uuid

import ray
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

from config import AppConfig
from data import DatasetManager, TransferQueueBroker
from inference.inference import VLLMRolloutWorker
from reward import HumanFeedback, RewardEnsemble
from train.train import DistributedDPOTrainLoop
from utils import MetricTracker, TrainControl


class RolloutProducer:
    def __init__(self, config, inference, control, queue):
        self.cfg = config
        self.inference = inference
        self.control = control
        self.queue = queue

    def run(self):
        data = DatasetManager(self.cfg.data)
        offset = 0
        while not ray.get(self.control.should_stop.remote()):
            if ray.get(self.queue.size.remote("rollout")) >= self.cfg.inference.max_rollout_queue:
                time.sleep(0.2)
                continue

            examples = data.examples(
                self.cfg.data.train_split,
                self.cfg.inference.prompts_per_batch,
                offset,
            )
            offset += len(examples)

            for group in ray.get(self.inference.generate.remote(examples)):
                ray.get(self.queue.put.remote(
                    "rollout",
                    group,
                    {"policy_version": group["policy_version"], "status": "ready"},
                ))


class TrainerJob:
    def run(self, config, inference, control, tracker, queue):
        trainer = TorchTrainer(
            train_loop_per_worker=DistributedDPOTrainLoop(),
            train_loop_config={
                "config": config,
                "inference": inference,
                "control": control,
                "tracker": tracker,
                "queue": queue,
            },
            scaling_config=ScalingConfig(num_workers=config.train.train_gpus, use_gpu=True,
                                         resources_per_worker={"GPU": 1, "CPU": config.ray.cpus_per_train_worker},
                                         placement_strategy=config.ray.placement_strategy),
        )
        return trainer.fit()


class RayOrchestrator:
    def __init__(self, config: AppConfig):
        self.cfg, self.actors = config, []
        self.stats = {"human": 0, "selected": 0, "all": 0, "neural_agree": 0,
                      "lgbm_agree": 0, "triple_agree": 0, "neural_only": 0, "lgbm_only": 0}

    def run(self):
        self._start_ray()
        self._check_resources()
        try:
            self._start_workers()
            self._baseline_eval()
            self.trainer_ref = self.trainer.run.remote(
                self.cfg, self.inference, self.control, self.tracker, self.queue
            )
            self.producer_ref = self.producer.run.remote()
            self._human_loop()
        finally:
            if hasattr(self, "control"): ray.get(self.control.request_stop.remote())
            for ref in (getattr(self, "trainer_ref", None), getattr(self, "producer_ref", None)):
                if ref is not None:
                    try: ray.get(ref, timeout=30)
                    except Exception: pass
            if hasattr(self, "tracker"):
                try: ray.get(self.tracker.close.remote())
                except Exception: pass
            if hasattr(self, "queue"):
                try:
                    ray.get(self.queue.close.remote())
                except Exception:
                    pass
            for actor in reversed(self.actors):
                try:
                    ray.kill(actor)
                except Exception:
                    pass

    def _start_ray(self):
        ray.init(address=self.cfg.ray.address) if self.cfg.ray.address else ray.init()
        print("Ray resources:", ray.cluster_resources())

    def _check_resources(self):
        need = self.cfg.train.train_gpus + self.cfg.inference.inference_gpus + self.cfg.reward.reward_gpus
        have = int(ray.cluster_resources().get("GPU", 0))
        if have < need: raise RuntimeError(f"Need {need} GPUs, Ray cluster has {have}")

    def _start_workers(self):
        run_id = self.cfg.queue.run_id or self.cfg.tracking.run_name or f"run-{uuid.uuid4().hex[:10]}"
        queue_cls = ray.remote(TransferQueueBroker).options(num_cpus=self.cfg.queue.broker_cpus)
        self.queue = queue_cls.remote(
            run_id,
            self.cfg.queue.rollout_id,
            self.cfg.queue.preference_id,
        )
        self.actors.append(self.queue)
        print("TransferQueue:", ray.get(self.queue.describe.remote()))

        self.control = ray.remote(TrainControl).remote(); self.actors.append(self.control)
        self.tracker = ray.remote(MetricTracker).remote(self.cfg.tracking); self.actors.append(self.tracker)
        self.inference = ray.remote(VLLMRolloutWorker).options(num_gpus=self.cfg.inference.inference_gpus,
                                                               num_cpus=self.cfg.ray.inference_cpus).remote(
            self.cfg.model, self.cfg.inference); self.actors.append(self.inference)
        reward_cls = ray.remote(RewardEnsemble).options(num_gpus=self.cfg.reward.reward_gpus,
                                                        num_cpus=self.cfg.ray.reward_cpus)
        self.reward = reward_cls.remote(self.cfg.model, self.cfg.reward); self.actors.append(self.reward)
        self.producer = ray.remote(RolloutProducer).remote(
            self.cfg, self.inference, self.control, self.queue
        ); self.actors.append(self.producer)
        self.trainer = ray.remote(TrainerJob).remote(); self.actors.append(self.trainer)
        print("Inference:", ray.get(self.inference.ready.remote()))

    def _baseline_eval(self):
        data = DatasetManager(self.cfg.data)
        self.eval_examples = data.examples(self.cfg.data.eval_split, self.cfg.data.eval_samples)
        self.base_score = ray.get(self.inference.evaluate.remote(self.eval_examples))
        self.last_eval_version = 0
        self.tracker.log.remote({"eval/base": self.base_score, "eval/by_optimizer_step": self.base_score,
                                 "eval/by_human_groups": self.base_score, "eval/by_used_pairs": self.base_score,
                                 "axis/optimizer_step": 0, "axis/human_groups": 0, "axis/used_pairs": 0})
        print(f"Baseline eval: {self.base_score:.4f}")

    def _check_trainer(self):
        """Fail fast if the asynchronous TorchTrainer has exited or crashed."""
        if not hasattr(self, "trainer_ref"):
            return False

        ready, _ = ray.wait([self.trainer_ref], timeout=0)
        if not ready:
            return False

        # Important: ray.get re-raises the real trainer exception here.
        result = ray.get(ready[0])
        print(f"[TRAIN] trainer finished: {result}", flush=True)
        return True

    def _human_loop(self):
        ui = HumanFeedback()
        while not ray.get(self.control.should_stop.remote()):
            if self._check_trainer():
                break
            groups, _ = ray.get(self.queue.pop.remote("rollout", 1))
            if not groups: time.sleep(0.2); self._maybe_eval(); continue
            group = groups[0]
            prediction = ray.get(self.reward.score.remote(group))
            human = ui.score(group)
            if human is None: break
            decision = ray.get(self.reward.decide.remote(prediction, human))
            self.reward.update.remote(group["id"], human)
            self._record(group, prediction, human, decision)
            if decision["selected"]:
                i, j = decision["top"], decision["bottom"]
                sample = {"id": group["id"], "prompt": group["prompt"], "chosen": group["responses"][i],
                          "rejected": group["responses"][j], "ref_chosen_logp": group["ref_logps"][i],
                          "ref_rejected_logp": group["ref_logps"][j], "policy_version": group["policy_version"]}
                ray.get(self.queue.put.remote(
                    "preference",
                    sample,
                    {"policy_version": group["policy_version"], "status": "ready"},
                ))
                self.stats["selected"] += 1
                preference_size = ray.get(self.queue.size.remote("preference"))
                print(
                    f"[QUEUE] selected={self.stats['selected']} "
                    f"preference_pending={preference_size}",
                    flush=True,
                )
            self._log_data(); self._maybe_eval()

    def _record(self, group, prediction, human, decision):
        self.stats["human"] += 1
        for k, v in decision["eligible"].items(): self.stats[k] += int(v)
        self.tracker.labeled.remote({**group, "human_scores": human, "neural_scores": prediction["neural"],
                                     "lgbm_scores": prediction["lgbm"], "decision": decision})

    def _log_data(self):
        n = max(self.stats["human"], 1)
        rollout_size, preference_size = ray.get([
            self.queue.size.remote("rollout"),
            self.queue.size.remote("preference"),
        ])
        metrics = {
            "axis/human_groups": self.stats["human"],
            "data/selected_pairs": self.stats["selected"],
            "data/selection_rate": self.stats["selected"] / n,
            "queue/rollout": rollout_size,
            "queue/preference": preference_size,
        }
        metrics.update({f"data/candidate_{k}": self.stats[k] for k in
                        ("all", "neural_agree", "lgbm_agree", "triple_agree", "neural_only", "lgbm_only")})
        self.tracker.log.remote(metrics)

    def _maybe_eval(self):
        status = ray.get(self.control.status.remote()); v = status["version"]
        if v <= self.last_eval_version or v % self.cfg.tracking.eval_every_versions: return
        score = ray.get(self.inference.evaluate.remote(self.eval_examples)); self.last_eval_version = v
        used = status["used_pairs"]; step = status["optimizer_step"]
        efficiency = (score - self.base_score) * 1000 / max(used, 1)
        self.tracker.log.remote({"model/policy_version": v, "eval/score": score,
                                 "eval/by_optimizer_step": score, "eval/by_human_groups": score,
                                 "eval/by_used_pairs": score, "efficiency/score_gain_per_1k_pairs": efficiency,
                                 "axis/optimizer_step": step, "axis/human_groups": self.stats["human"],
                                 "axis/used_pairs": used})
        print(f"Eval v{v}: {score:.4f} | used_pairs={used} | selection={self.stats['selected']}/{self.stats['human']}")
