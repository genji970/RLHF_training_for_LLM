import argparse
import os

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "bfloat16"
    trust_remote_code: bool = False


@dataclass
class DataConfig:
    dataset_name: str = "allenai/openbookqa"

    # HuggingFace dataset config/subset
    # ex) OpenBookQA -> main
    # ex) ARC -> ARC-Challenge
    dataset_subset: Optional[str] = "main"

    dataset_split: str = "train"

    # auto / conversation / mcqa / qa
    task_type: str = "auto"

    max_samples: Optional[int] = None

    cache_dir: Optional[str] = None

    dataloader_workers: int = 2


@dataclass
class TrainConfig:
    train_gpus: int = 3

    batch_size: int = 2
    epochs: int = 3

    learning_rate: float = 3e-5
    weight_decay: float = 0.01

    max_length: int = 512


@dataclass
class InferenceConfig:
    inference_gpus: int = 1

    rollout_batch_size: int = 12

    max_new_tokens: int = 128

    temperature: float = 0.8
    top_p: float = 0.95

    gpu_memory_utilization: float = 0.9


@dataclass
class RayConfig:
    address: Optional[str] = None

    cpus_per_train_worker: int = 2
    inference_cpus: int = 2

    placement_strategy: str = "PACK"


@dataclass
class AppConfig:
    model: ModelConfig = field(
        default_factory=ModelConfig
    )

    data: DataConfig = field(
        default_factory=DataConfig
    )

    train: TrainConfig = field(
        default_factory=TrainConfig
    )

    inference: InferenceConfig = field(
        default_factory=InferenceConfig
    )

    ray: RayConfig = field(
        default_factory=RayConfig
    )

    def validate(self):

        if self.train.train_gpus < 1:
            raise ValueError(
                "train_gpus must be >= 1"
            )

        if self.inference.inference_gpus < 1:
            raise ValueError(
                "inference_gpus must be >= 1"
            )

        if (
            self.inference.rollout_batch_size
            % self.train.train_gpus
            != 0
        ):
            raise ValueError(
                "rollout_batch_size must be divisible "
                "by train_gpus."
            )

        if self.data.task_type not in {
            "auto",
            "conversation",
            "mcqa",
            "qa",
        }:
            raise ValueError(
                f"Unknown task_type: "
                f"{self.data.task_type}"
            )


class ConfigParser:

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self._add_model_args()
        self._add_data_args()
        self._add_train_args()
        self._add_inference_args()
        self._add_ray_args()

    def _add_model_args(self):

        self.parser.add_argument(
            "--model-name",
            type=str,
            default="Qwen/Qwen2.5-0.5B-Instruct",
        )

        self.parser.add_argument(
            "--dtype",
            type=str,
            choices=[
                "auto",
                "float16",
                "bfloat16",
                "float32",
            ],
            default="bfloat16",
        )

        self.parser.add_argument(
            "--trust-remote-code",
            action=argparse.BooleanOptionalAction,
            default=False,
        )

    def _add_data_args(self):

        self.parser.add_argument(
            "--dataset-name",
            type=str,
            default="allenai/openbookqa",
        )

        self.parser.add_argument(
            "--dataset-subset",
            type=str,
            default="main",
        )

        self.parser.add_argument(
            "--dataset-split",
            type=str,
            default="train",
        )

        self.parser.add_argument(
            "--task-type",
            type=str,
            choices=[
                "auto",
                "conversation",
                "mcqa",
                "qa",
            ],
            default="auto",
        )

        self.parser.add_argument(
            "--max-samples",
            type=int,
            default=None,
        )

        self.parser.add_argument(
            "--cache-dir",
            type=str,
            default=None,
        )

        self.parser.add_argument(
            "--dataloader-workers",
            type=int,
            default=2,
        )

    def _add_train_args(self):

        self.parser.add_argument(
            "--train-gpus",
            type=int,
            default=3,
        )

        self.parser.add_argument(
            "--batch-size",
            type=int,
            default=2,
        )

        self.parser.add_argument(
            "--epochs",
            type=int,
            default=3,
        )

        self.parser.add_argument(
            "--learning-rate",
            type=float,
            default=3e-5,
        )

        self.parser.add_argument(
            "--weight-decay",
            type=float,
            default=0.01,
        )

        self.parser.add_argument(
            "--max-length",
            type=int,
            default=512,
        )

    def _add_inference_args(self):

        self.parser.add_argument(
            "--inference-gpus",
            type=int,
            default=1,
        )

        self.parser.add_argument(
            "--rollout-batch-size",
            type=int,
            default=12,
        )

        self.parser.add_argument(
            "--max-new-tokens",
            type=int,
            default=128,
        )

        self.parser.add_argument(
            "--temperature",
            type=float,
            default=0.8,
        )

        self.parser.add_argument(
            "--top-p",
            type=float,
            default=0.95,
        )

        self.parser.add_argument(
            "--gpu-memory-utilization",
            type=float,
            default=0.9,
        )

    def _add_ray_args(self):

        self.parser.add_argument(
            "--ray-address",
            type=str,
            default=os.environ.get(
                "RAY_ADDRESS"
            ),
        )

        self.parser.add_argument(
            "--cpus-per-train-worker",
            type=int,
            default=2,
        )

        self.parser.add_argument(
            "--inference-cpus",
            type=int,
            default=2,
        )

        self.parser.add_argument(
            "--placement-strategy",
            type=str,
            choices=[
                "PACK",
                "SPREAD",
                "STRICT_PACK",
                "STRICT_SPREAD",
            ],
            default="PACK",
        )

    def parse(self) -> AppConfig:

        args = self.parser.parse_args()

        subset = args.dataset_subset

        if subset in {
            "",
            "none",
            "None",
            "null",
        }:
            subset = None

        config = AppConfig(
            model=ModelConfig(
                model_name=args.model_name,
                dtype=args.dtype,
                trust_remote_code=(
                    args.trust_remote_code
                ),
            ),
            data=DataConfig(
                dataset_name=args.dataset_name,
                dataset_subset=subset,
                dataset_split=args.dataset_split,
                task_type=args.task_type,
                max_samples=args.max_samples,
                cache_dir=args.cache_dir,
                dataloader_workers=(
                    args.dataloader_workers
                ),
            ),
            train=TrainConfig(
                train_gpus=args.train_gpus,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=(
                    args.learning_rate
                ),
                weight_decay=args.weight_decay,
                max_length=args.max_length,
            ),
            inference=InferenceConfig(
                inference_gpus=(
                    args.inference_gpus
                ),
                rollout_batch_size=(
                    args.rollout_batch_size
                ),
                max_new_tokens=(
                    args.max_new_tokens
                ),
                temperature=args.temperature,
                top_p=args.top_p,
                gpu_memory_utilization=(
                    args.gpu_memory_utilization
                ),
            ),
            ray=RayConfig(
                address=args.ray_address,
                cpus_per_train_worker=(
                    args.cpus_per_train_worker
                ),
                inference_cpus=(
                    args.inference_cpus
                ),
                placement_strategy=(
                    args.placement_strategy
                ),
            ),
        )

        config.validate()

        return config