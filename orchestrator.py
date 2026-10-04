import os
import time
import uuid

import ray
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

from config import AppConfig
from data import DatasetManager, TransferQueueBroker
from debug import Debugger
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
        self.debug = Debugger(config.debug, "inference")

    def run(self):
        self.debug.environment("inference")
        data = DatasetManager(self.cfg.data, self.cfg.debug)
        offset = 0
        last_backpressure_log = 0.0
        last_backpressure_size = None
        while not ray.get(self.control.should_stop.remote()):
            # This is a hot polling path. Do not wrap it in a debug stage: BEGIN/END
            # lines every 200 ms can bury the interactive HumanFeedback input prompt.
            rollout_size = ray.get(self.queue.size.remote("rollout"))
            if rollout_size >= self.cfg.inference.max_rollout_queue:
                now = time.monotonic()
                if (
                    rollout_size != last_backpressure_size
                    or now - last_backpressure_log >= 5.0
                ):
                    self.debug.log(
                        "inference",
                        "rollout_backpressure",
                        queue_size=rollout_size,
                        max_queue=self.cfg.inference.max_rollout_queue,
                    )
                    last_backpressure_log = now
                    last_backpressure_size = rollout_size
                time.sleep(0.2)
                continue

            last_backpressure_size = None

            with self.debug.stage(
                "data",
                "rollout_examples",
                split=self.cfg.data.train_split,
                offset=offset,
                count=self.cfg.inference.prompts_per_batch,
            ):
                examples = data.examples(
                    self.cfg.data.train_split,
                    self.cfg.inference.prompts_per_batch,
                    offset,
                )
            offset += len(examples)

            with self.debug.stage("inference", "vllm_generate", prompts=len(examples)):
                groups = ray.get(self.inference.generate.remote(examples))
            self.debug.log("inference", "generation_done", groups=len(groups))

            for group in groups:
                with self.debug.stage(
                    "queue",
                    "rollout_put",
                    group_id=group.get("id"),
                    policy_version=group.get("policy_version"),
                ):
                    ray.get(self.queue.put.remote(
                        "rollout",
                        group,
                        {"policy_version": group["policy_version"], "status": "ready"},
                    ))


class TrainerJob:
    def run(self, config, inference, control, tracker, queue):
        debug = Debugger(config.debug, "ray")
        debug.log(
            "ray",
            "torch_trainer_create",
            train_gpus=config.train.train_gpus,
            placement_strategy=config.ray.placement_strategy,
            cpus_per_worker=config.ray.cpus_per_train_worker,
        )
        with debug.stage("ray", "torch_trainer_init"):
            trainer = TorchTrainer(
                train_loop_per_worker=DistributedDPOTrainLoop(),
                train_loop_config={
                    "config": config,
                    "inference": inference,
                    "control": control,
                    "tracker": tracker,
                    "queue": queue,
                },
                scaling_config=ScalingConfig(
                    num_workers=config.train.train_gpus,
                    use_gpu=True,
                    resources_per_worker={
                        "GPU": 1,
                        "CPU": config.ray.cpus_per_train_worker,
                    },
                    placement_strategy=config.ray.placement_strategy,
                ),
            )
        with debug.stage("ray", "torch_trainer_fit"):
            result = trainer.fit()
        debug.log("ray", "torch_trainer_finished", result=result)
        return result


class RayOrchestrator:
    def __init__(self, config: AppConfig):
        self.cfg, self.actors = config, []
        self.debug = Debugger(config.debug, "ray")
        self.stats = {
            "human": 0,
            "selected": 0,
            "all": 0,
            "neural_agree": 0,
            "lgbm_agree": 0,
            "triple_agree": 0,
            "neural_only": 0,
            "lgbm_only": 0,
        }

    def run(self):
        self.debug.log(
            "ray",
            "application_start",
            debug_all=self.cfg.debug.all,
            debug_train=self.cfg.debug.train,
            debug_fsdp=self.cfg.debug.fsdp,
            debug_queue=self.cfg.debug.queue,
            debug_ray=self.cfg.debug.ray,
            debug_reward=self.cfg.debug.reward,
            debug_inference=self.cfg.debug.inference,
            debug_data=self.cfg.debug.data,
        )
        self._start_ray()
        self._check_resources()
        try:
            self._start_workers()
            self._baseline_eval()
            with self.debug.stage("ray", "launch_trainer_actor"):
                self.trainer_ref = self.trainer.run.remote(
                    self.cfg, self.inference, self.control, self.tracker, self.queue
                )
            with self.debug.stage("ray", "launch_rollout_producer"):
                self.producer_ref = self.producer.run.remote()
            self._human_loop()
        finally:
            self.debug.log("ray", "shutdown_begin")
            if hasattr(self, "control"):
                try:
                    ray.get(self.control.request_stop.remote())
                except Exception as exc:
                    self.debug.exception("ray", "request_stop_failed", exc)

            for name, ref in (
                ("trainer", getattr(self, "trainer_ref", None)),
                ("producer", getattr(self, "producer_ref", None)),
            ):
                if ref is not None:
                    try:
                        with self.debug.stage("ray", f"join_{name}"):
                            ray.get(ref, timeout=30)
                    except Exception as exc:
                        self.debug.exception("ray", f"join_{name}_failed", exc)

            if hasattr(self, "tracker"):
                try:
                    ray.get(self.tracker.close.remote())
                except Exception as exc:
                    self.debug.exception("ray", "tracker_close_failed", exc)

            if hasattr(self, "queue"):
                try:
                    ray.get(self.queue.close.remote())
                except Exception as exc:
                    self.debug.exception("queue", "queue_close_failed", exc)

            for actor in reversed(self.actors):
                try:
                    ray.kill(actor)
                except Exception as exc:
                    self.debug.exception("ray", "actor_kill_failed", exc)
            self.debug.log("ray", "shutdown_end")

    def _start_ray(self):
        # Keep NCCL enabled, but disable the CUDA GPU-to-GPU P2P transport on
        # this machine. NCCL will use its normal fallback transport (typically
        # shared memory for same-node GPUs). This is the only workaround kept
        # from the FSDP/NCCL debugging session.
        os.environ["NCCL_P2P_DISABLE"] = "1"
        init_kwargs = {
            "runtime_env": {"env_vars": {"NCCL_P2P_DISABLE": "1"}},
        }
        if self.cfg.ray.address:
            init_kwargs["address"] = self.cfg.ray.address

        with self.debug.stage("ray", "ray_init", address=self.cfg.ray.address):
            ray.init(**init_kwargs)
        resources = ray.cluster_resources()
        self.debug.log("ray", "cluster_resources", resources=resources)
        print("Ray resources:", resources)

    def _check_resources(self):
        need = (
            self.cfg.train.train_gpus
            + self.cfg.inference.inference_gpus
            + self.cfg.reward.reward_gpus
        )
        have = int(ray.cluster_resources().get("GPU", 0))
        self.debug.log("ray", "resource_check", gpu_needed=need, gpu_available=have)
        if have < need:
            raise RuntimeError(f"Need {need} GPUs, Ray cluster has {have}")

    def _start_workers(self):
        run_id = (
            self.cfg.queue.run_id
            or self.cfg.tracking.run_name
            or f"run-{uuid.uuid4().hex[:10]}"
        )
        self.debug.log("ray", "run_id", run_id=run_id)

        with self.debug.stage("ray", "create_queue_actor"):
            queue_cls = ray.remote(TransferQueueBroker).options(
                num_cpus=self.cfg.queue.broker_cpus
            )
            self.queue = queue_cls.remote(
                run_id,
                self.cfg.queue.rollout_id,
                self.cfg.queue.preference_id,
                self.cfg.debug,
            )
        self.actors.append(self.queue)
        with self.debug.stage("queue", "queue_describe"):
            description = ray.get(self.queue.describe.remote())
        print("TransferQueue:", description)

        with self.debug.stage("ray", "create_control_actor"):
            self.control = ray.remote(TrainControl).remote()
        self.actors.append(self.control)

        with self.debug.stage("ray", "create_tracker_actor"):
            self.tracker = ray.remote(MetricTracker).remote(self.cfg.tracking)
        self.actors.append(self.tracker)

        with self.debug.stage(
            "ray",
            "create_inference_actor",
            gpus=self.cfg.inference.inference_gpus,
        ):
            self.inference = ray.remote(VLLMRolloutWorker).options(
                num_gpus=self.cfg.inference.inference_gpus,
                num_cpus=self.cfg.ray.inference_cpus,
            ).remote(self.cfg.model, self.cfg.inference)
        self.actors.append(self.inference)

        with self.debug.stage(
            "ray",
            "create_reward_actor",
            gpus=self.cfg.reward.reward_gpus,
        ):
            reward_cls = ray.remote(RewardEnsemble).options(
                num_gpus=self.cfg.reward.reward_gpus,
                num_cpus=self.cfg.ray.reward_cpus,
            )
            self.reward = reward_cls.remote(
                self.cfg.model,
                self.cfg.reward,
                self.cfg.debug,
            )
        self.actors.append(self.reward)

        with self.debug.stage("ray", "create_rollout_actor"):
            self.producer = ray.remote(RolloutProducer).remote(
                self.cfg, self.inference, self.control, self.queue
            )
        self.actors.append(self.producer)

        with self.debug.stage("ray", "create_trainer_actor"):
            self.trainer = ray.remote(TrainerJob).remote()
        self.actors.append(self.trainer)

        with self.debug.stage("inference", "inference_ready"):
            ready = ray.get(self.inference.ready.remote())
        self.debug.log("inference", "inference_ready_result", result=ready)
        print("Inference:", ready)

    def _baseline_eval(self):
        data = DatasetManager(self.cfg.data, self.cfg.debug)
        with self.debug.stage("data", "baseline_examples", count=self.cfg.data.eval_samples):
            self.eval_examples = data.examples(
                self.cfg.data.eval_split,
                self.cfg.data.eval_samples,
            )
        with self.debug.stage("inference", "baseline_eval", count=len(self.eval_examples)):
            self.base_score = ray.get(self.inference.evaluate.remote(self.eval_examples))

        self.last_eval_version = 0
        self.tracker.log.remote({
            "eval/base": self.base_score,
            "eval/by_optimizer_step": self.base_score,
            "eval/by_human_groups": self.base_score,
            "eval/by_used_pairs": self.base_score,
            "axis/optimizer_step": 0,
            "axis/human_groups": 0,
            "axis/used_pairs": 0,
        })
        self.debug.log("inference", "baseline_eval_done", score=self.base_score)
        print(f"Baseline eval: {self.base_score:.4f}")

    def _check_trainer(self):
        if not hasattr(self, "trainer_ref"):
            return False

        ready, _ = ray.wait([self.trainer_ref], timeout=0)
        if not ready:
            return False

        # ray.get intentionally re-raises the real trainer exception here.
        self.debug.log("ray", "trainer_ref_ready")
        result = ray.get(ready[0])
        self.debug.log("ray", "trainer_finished", result=result)
        print(f"[TRAIN] trainer finished: {result}", flush=True)
        return True

    def _human_loop(self):
        ui = HumanFeedback()
        self.debug.log("ray", "human_loop_start")
        while not ray.get(self.control.should_stop.remote()):
            if self._check_trainer():
                break

            with self.debug.stage("queue", "human_pop_rollout"):
                groups, _ = ray.get(self.queue.pop.remote("rollout", 1))
            if not groups:
                time.sleep(0.2)
                self._maybe_eval()
                continue

            group = groups[0]
            gid = group.get("id")
            self.debug.log("queue", "human_group_received", group_id=gid)

            # Start reward inference asynchronously, but DO NOT wait for it before
            # showing the interactive human prompt. This keeps the terminal usable
            # even if the reward model is slow (e.g. CPU reward_gpus=0).
            prediction_ref = self.reward.score.remote(group)
            self.debug.log("reward", "reward_score_submitted", group_id=gid)
            self.debug.log("ray", "human_prompt_begin", group_id=gid)

            human = ui.score(group)
            if human is None:
                self.debug.log("ray", "human_quit")
                try:
                    ray.cancel(prediction_ref, force=False)
                except Exception:
                    pass
                break

            self.debug.log("ray", "human_prompt_end", group_id=gid)
            with self.debug.stage("reward", "reward_score_wait", group_id=gid):
                prediction = ray.get(prediction_ref)

            with self.debug.stage("reward", "reward_decide", group_id=gid):
                decision = ray.get(self.reward.decide.remote(prediction, human))
            with self.debug.stage("reward", "reward_update", group_id=gid):
                self.reward.update.remote(group["id"], human)

            self._record(group, prediction, human, decision)
            if decision["selected"]:
                i, j = decision["top"], decision["bottom"]
                sample = {
                    "id": group["id"],
                    "prompt": group["prompt"],
                    "chosen": group["responses"][i],
                    "rejected": group["responses"][j],
                    "ref_chosen_logp": group["ref_logps"][i],
                    "ref_rejected_logp": group["ref_logps"][j],
                    "policy_version": group["policy_version"],
                }
                with self.debug.stage("queue", "human_put_preference", group_id=gid):
                    ray.get(self.queue.put.remote(
                        "preference",
                        sample,
                        {"policy_version": group["policy_version"], "status": "ready"},
                    ))
                self.stats["selected"] += 1
                preference_size = ray.get(self.queue.size.remote("preference"))
                self.debug.log(
                    "queue",
                    "preference_selected",
                    selected=self.stats["selected"],
                    preference_pending=preference_size,
                    group_id=gid,
                )

            self._log_data()
            self._maybe_eval()

        self.debug.log("ray", "human_loop_end")

    def _record(self, group, prediction, human, decision):
        self.stats["human"] += 1
        for k, v in decision["eligible"].items():
            self.stats[k] += int(v)
        self.tracker.labeled.remote({
            **group,
            "human_scores": human,
            "neural_scores": prediction["neural"],
            "lgbm_scores": prediction["lgbm"],
            "decision": decision,
        })

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
        metrics.update({
            f"data/candidate_{k}": self.stats[k]
            for k in (
                "all",
                "neural_agree",
                "lgbm_agree",
                "triple_agree",
                "neural_only",
                "lgbm_only",
            )
        })
        self.tracker.log.remote(metrics)
        self.debug.log("queue", "pipeline_metrics", **metrics)

    def _maybe_eval(self):
        status = ray.get(self.control.status.remote())
        v = status["version"]
        if v <= self.last_eval_version or v % self.cfg.tracking.eval_every_versions:
            return

        with self.debug.stage("inference", "version_eval", version=v):
            score = ray.get(self.inference.evaluate.remote(self.eval_examples))
        self.last_eval_version = v
        used = status["used_pairs"]
        step = status["optimizer_step"]
        efficiency = (score - self.base_score) * 1000 / max(used, 1)
        self.tracker.log.remote({
            "model/policy_version": v,
            "eval/score": score,
            "eval/by_optimizer_step": score,
            "eval/by_human_groups": score,
            "eval/by_used_pairs": score,
            "efficiency/score_gain_per_1k_pairs": efficiency,
            "axis/optimizer_step": step,
            "axis/human_groups": self.stats["human"],
            "axis/used_pairs": used,
        })
        self.debug.log(
            "inference",
            "version_eval_done",
            version=v,
            score=score,
            used_pairs=used,
            optimizer_step=step,
        )
        print(
            f"Eval v{v}: {score:.4f} | used_pairs={used} | "
            f"selection={self.stats['selected']}/{self.stats['human']}"
        )
