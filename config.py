import argparse
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    policy_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    reward_name: Optional[str] = None
    dtype: str = "bfloat16"
    trust_remote_code: bool = False


@dataclass
class DataConfig:
    dataset_name: str = "allenai/openbookqa"
    dataset_subset: Optional[str] = "main"
    train_split: str = "train"
    eval_split: str = "validation"
    task_type: str = "auto"
    max_samples: Optional[int] = None
    eval_samples: int = 128
    cache_dir: Optional[str] = None


@dataclass
class TrainConfig:
    train_gpus: int = 2
    batch_size: int = 4
    max_length: int = 512
    learning_rate: float = 3e-6
    weight_decay: float = 0.01
    beta: float = 0.1
    max_steps: int = 1000
    sync_every: int = 4
    max_policy_lag: int = 3
    poll_seconds: float = 0.5
    gradient_checkpointing: bool = True
    checkpoint_dir: str = "./checkpoints"


@dataclass
class InferenceConfig:
    inference_gpus: int = 1
    prompts_per_batch: int = 2
    n_responses: int = 4
    max_rollout_queue: int = 8
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    gpu_memory_utilization: float = 0.85


@dataclass
class RewardConfig:
    reward_gpus: int = 1
    feature_dim: int = 128
    learning_rate: float = 1e-3
    train_steps: int = 2
    warmup_groups: int = 32
    lgbm_refit_every: int = 8
    filter_mode: str = "triple_agree"


@dataclass
class QueueConfig:
    # One TransferQueue runtime per experiment.
    # Logical queues are isolated with unique partition IDs:
    #   <run_id>:<rollout_id>
    #   <run_id>:<preference_id>
    run_id: Optional[str] = None
    rollout_id: str = "rollout"
    preference_id: str = "preference"
    broker_cpus: float = 1.0


@dataclass
class RayConfig:
    address: Optional[str] = None
    cpus_per_train_worker: int = 2
    inference_cpus: int = 2
    reward_cpus: int = 2
    placement_strategy: str = "PACK"


@dataclass
class TrackingConfig:
    output_dir: str = "./runs"
    wandb: bool = True
    project: str = "self-reward"
    run_name: Optional[str] = None
    eval_every_versions: int = 1


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    ray: RayConfig = field(default_factory=RayConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def validate(self):
        if self.model.dtype != "bfloat16":
            raise ValueError("This configuration is BF16-only: --dtype must be bfloat16")
        if self.train.train_gpus < 1 or self.inference.inference_gpus < 1:
            raise ValueError("train_gpus and inference_gpus must be >= 1")
        if self.reward.reward_gpus < 0:
            raise ValueError("reward_gpus must be >= 0")
        if self.inference.n_responses < 2:
            raise ValueError("n_responses must be >= 2")
        if self.train.batch_size % self.train.train_gpus:
            raise ValueError("batch_size must be divisible by train_gpus")
        valid = {"all", "neural_agree", "lgbm_agree", "triple_agree", "neural_only", "lgbm_only"}
        if self.reward.filter_mode not in valid:
            raise ValueError(f"filter_mode must be one of {sorted(valid)}")
        if self.data.task_type not in {"auto", "conversation", "mcqa", "qa"}:
            raise ValueError("task_type must be auto/conversation/mcqa/qa")


class ConfigParser:
    def __init__(self):
        p = self.p = argparse.ArgumentParser()
        self._add("--policy-name", str, ModelConfig.policy_name)
        self._add("--reward-name", str, None)
        self._add("--dtype", str, ModelConfig.dtype)
        p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)

        self._add("--dataset-name", str, DataConfig.dataset_name)
        self._add("--dataset-subset", str, DataConfig.dataset_subset)
        self._add("--train-split", str, DataConfig.train_split)
        self._add("--eval-split", str, DataConfig.eval_split)
        self._add("--task-type", str, DataConfig.task_type)
        self._add("--max-samples", int, None)
        self._add("--eval-samples", int, DataConfig.eval_samples)

        self._add("--train-gpus", int, TrainConfig.train_gpus)
        self._add("--batch-size", int, TrainConfig.batch_size)
        self._add("--max-length", int, TrainConfig.max_length)
        self._add("--learning-rate", float, TrainConfig.learning_rate)
        self._add("--beta", float, TrainConfig.beta)
        self._add("--max-steps", int, TrainConfig.max_steps)
        self._add("--sync-every", int, TrainConfig.sync_every)
        self._add("--max-policy-lag", int, TrainConfig.max_policy_lag)
        self._add("--checkpoint-dir", str, TrainConfig.checkpoint_dir)

        self._add("--inference-gpus", int, InferenceConfig.inference_gpus)
        self._add("--prompts-per-batch", int, InferenceConfig.prompts_per_batch)
        self._add("--n-responses", int, InferenceConfig.n_responses)
        self._add("--max-rollout-queue", int, InferenceConfig.max_rollout_queue)
        self._add("--max-new-tokens", int, InferenceConfig.max_new_tokens)
        self._add("--temperature", float, InferenceConfig.temperature)
        self._add("--top-p", float, InferenceConfig.top_p)
        self._add("--gpu-memory-utilization", float, InferenceConfig.gpu_memory_utilization)

        self._add("--reward-gpus", int, RewardConfig.reward_gpus)
        self._add("--feature-dim", int, RewardConfig.feature_dim)
        self._add("--reward-lr", float, RewardConfig.learning_rate)
        self._add("--reward-warmup-groups", int, RewardConfig.warmup_groups)
        self._add("--lgbm-refit-every", int, RewardConfig.lgbm_refit_every)
        self._add("--filter-mode", str, RewardConfig.filter_mode)

        self._add("--queue-run-id", str, None)
        self._add("--rollout-queue-id", str, QueueConfig.rollout_id)
        self._add("--preference-queue-id", str, QueueConfig.preference_id)

        self._add("--ray-address", str, os.environ.get("RAY_ADDRESS"))
        self._add("--placement-strategy", str, RayConfig.placement_strategy)
        self._add("--output-dir", str, TrackingConfig.output_dir)
        self._add("--wandb-project", str, TrackingConfig.project)
        self._add("--run-name", str, None)
        self._add("--eval-every-versions", int, TrackingConfig.eval_every_versions)
        p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)

    def _add(self, name, type_, default):
        self.p.add_argument(name, type=type_, default=default)

    @staticmethod
    def _none(value):
        return None if value in {None, "", "none", "None", "null"} else value

    def parse(self) -> AppConfig:
        a = self.p.parse_args()
        cfg = AppConfig(
            model=ModelConfig(a.policy_name, self._none(a.reward_name), a.dtype, a.trust_remote_code),
            data=DataConfig(a.dataset_name, self._none(a.dataset_subset), a.train_split, a.eval_split,
                            a.task_type, a.max_samples, a.eval_samples),
            train=TrainConfig(a.train_gpus, a.batch_size, a.max_length, a.learning_rate,
                              TrainConfig.weight_decay, a.beta, a.max_steps, a.sync_every,
                              a.max_policy_lag, TrainConfig.poll_seconds,
                              TrainConfig.gradient_checkpointing, a.checkpoint_dir),
            inference=InferenceConfig(a.inference_gpus, a.prompts_per_batch, a.n_responses,
                                      a.max_rollout_queue, a.max_new_tokens, a.temperature,
                                      a.top_p, a.gpu_memory_utilization),
            reward=RewardConfig(a.reward_gpus, a.feature_dim, a.reward_lr, RewardConfig.train_steps,
                                a.reward_warmup_groups, a.lgbm_refit_every, a.filter_mode),
            queue=QueueConfig(self._none(a.queue_run_id), a.rollout_queue_id,
                              a.preference_queue_id, QueueConfig.broker_cpus),
            ray=RayConfig(a.ray_address, RayConfig.cpus_per_train_worker, RayConfig.inference_cpus,
                          RayConfig.reward_cpus, a.placement_strategy),
            tracking=TrackingConfig(a.output_dir, a.wandb, a.wandb_project,
                                    a.run_name, a.eval_every_versions),
        )
        cfg.validate()
        return cfg
