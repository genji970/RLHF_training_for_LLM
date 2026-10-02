from typing import Dict

import ray

import torch
import torch.distributed as dist

from torch.distributed.fsdp import (
    ShardingStrategy,
)

from ray import train

from ray.train.torch import (
    prepare_model,
    prepare_data_loader,
    get_device,
)

from config import AppConfig

from model_load import ModelLoader

from data_load import (
    HFDatasetManager,
    SFTCollator,
)


class DistributedTrainLoop:

    def __call__(
        self,
        train_loop_config: Dict,
    ):

        config: AppConfig = (
            train_loop_config[
                "config"
            ]
        )

        inference_worker = (
            train_loop_config[
                "inference_worker"
            ]
        )

        worker = FSDPTrainerWorker(
            config=config,
            inference_worker=(
                inference_worker
            ),
        )

        worker.run()


class FSDPTrainerWorker:

    def __init__(
        self,
        config: AppConfig,
        inference_worker,
    ):

        self.config = config

        self.inference_worker = (
            inference_worker
        )

        self.context = (
            train.get_context()
        )

        self.rank = (
            self.context
            .get_world_rank()
        )

        self.world_size = (
            self.context
            .get_world_size()
        )

        self.device = get_device()

        self.model = None
        self.optimizer = None

        self.tokenizer = None

        self.dataset_manager = None

        self.dataloader = None

        self.generated_collator = None

    def run(self):

        self._setup()

        for epoch in range(
            self.config.train.epochs
        ):

            self._set_epoch(epoch)

            original_loss = (
                self._train_original_data()
            )

            generated_loss = (
                self._generate_and_train(
                    epoch
                )
            )

            original_loss = (
                self._distributed_mean(
                    original_loss
                )
            )

            generated_loss = (
                self._distributed_mean(
                    generated_loss
                )
            )

            if self.rank == 0:

                print(
                    f"epoch={epoch} "
                    f"original_loss="
                    f"{original_loss:.4f} "
                    f"generated_loss="
                    f"{generated_loss:.4f}"
                )

            train.report({
                "epoch": epoch,

                "original_loss":
                    original_loss,

                "generated_loss":
                    generated_loss,
            })

    def _setup(self):

        print(
            f"[rank {self.rank}] "
            f"world_size="
            f"{self.world_size}"
        )

        model_loader = ModelLoader(
            self.config.model
        )

        self.tokenizer = (
            model_loader
            .load_tokenizer()
        )

        self.model = (
            model_loader
            .load_train_model()
        )

        self.model = prepare_model(
            self.model,

            parallel_strategy="fsdp",

            parallel_strategy_kwargs={
                "sharding_strategy":
                    ShardingStrategy
                    .FULL_SHARD,

                "use_orig_params":
                    True,
            },
        )

        self.optimizer = (
            torch.optim.AdamW(
                self.model.parameters(),

                lr=(
                    self.config.train
                    .learning_rate
                ),

                weight_decay=(
                    self.config.train
                    .weight_decay
                ),
            )
        )

        self.dataset_manager = (
            HFDatasetManager(
                config=self.config.data,

                tokenizer=(
                    self.tokenizer
                ),

                batch_size=(
                    self.config.train
                    .batch_size
                ),

                max_length=(
                    self.config.train
                    .max_length
                ),
            )
        )

        self.dataset_manager.load()

        self.dataloader = (
            self.dataset_manager
            .create_dataloader()
        )

        self.dataloader = (
            prepare_data_loader(
                self.dataloader
            )
        )

        self.generated_collator = (
            SFTCollator(
                tokenizer=(
                    self.tokenizer
                ),

                max_length=(
                    self.config.train
                    .max_length
                ),
            )
        )

    def _set_epoch(
        self,
        epoch,
    ):

        sampler = getattr(
            self.dataloader,
            "sampler",
            None,
        )

        if (
            sampler is not None
            and hasattr(
                sampler,
                "set_epoch",
            )
        ):
            sampler.set_epoch(
                epoch
            )

    def _train_original_data(self):

        self.model.train()

        total_loss = 0.0
        steps = 0

        for batch in self.dataloader:

            loss = self._train_batch(
                batch
            )

            total_loss += loss
            steps += 1

        return (
            total_loss
            / max(steps, 1)
        )

    def _generate_and_train(
        self,
        epoch,
    ):

        generated_examples = None

        rollout_count = (
            self.config
            .inference
            .rollout_batch_size
        )

        if self.rank == 0:

            rollout_examples = (
                self.dataset_manager
                .get_rollout_examples(
                    count=rollout_count,

                    offset=(
                        epoch
                        * rollout_count
                    ),
                )
            )

            generated_examples = (
                ray.get(
                    self.inference_worker
                    .generate
                    .remote(
                        rollout_examples
                    )
                )
            )

        payload = [
            generated_examples
        ]

        if self.world_size > 1:

            dist.broadcast_object_list(
                payload,
                src=0,
            )

        generated_examples = (
            payload[0]
        )

        local_examples = (
            self._shard_generated_data(
                generated_examples
            )
        )

        return (
            self._train_generated_data(
                local_examples
            )
        )

    def _shard_generated_data(
        self,
        examples,
    ):

        if (
            len(examples)
            % self.world_size
            != 0
        ):
            raise ValueError(
                "Generated example count "
                "must be divisible by "
                "world_size."
            )

        per_rank = (
            len(examples)
            // self.world_size
        )

        start = (
            self.rank
            * per_rank
        )

        end = (
            start
            + per_rank
        )

        return examples[
            start:end
        ]

    def _train_generated_data(
        self,
        examples,
    ):

        batch_size = (
            self.config.train.batch_size
        )

        total_loss = 0.0
        steps = 0

        for start in range(
            0,
            len(examples),
            batch_size,
        ):

            batch_examples = examples[
                start:
                start + batch_size
            ]

            batch = (
                self.generated_collator(
                    batch_examples
                )
            )

            batch = (
                self._move_to_device(
                    batch
                )
            )

            loss = self._train_batch(
                batch
            )

            total_loss += loss
            steps += 1

        return (
            total_loss
            / max(steps, 1)
        )

    def _train_batch(
        self,
        batch,
    ):

        self.optimizer.zero_grad()

        outputs = self.model(
            input_ids=(
                batch["input_ids"]
            ),

            attention_mask=(
                batch[
                    "attention_mask"
                ]
            ),

            labels=(
                batch["labels"]
            ),
        )

        loss = outputs.loss

        loss.backward()

        self.optimizer.step()

        return (
            loss
            .detach()
            .float()
            .item()
        )

    def _move_to_device(
        self,
        batch,
    ):

        return {
            key:
                value.to(
                    self.device
                )

            for key, value
            in batch.items()
        }

    def _distributed_mean(
        self,
        value,
    ):

        tensor = torch.tensor(
            value,
            dtype=torch.float32,
            device=self.device,
        )

        if self.world_size > 1:

            dist.all_reduce(
                tensor,
                op=dist.ReduceOp.SUM,
            )

            tensor /= (
                self.world_size
            )

        return tensor.item()