import functools
import time
from contextlib import nullcontext
from pathlib import Path

import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from ray import train
from ray.train.torch import get_device, prepare_model

from config import AppConfig
from data import DPOCollator
from transformers import AutoModelForCausalLM, AutoTokenizer


class DistributedDPOTrainLoop:
    def __call__(self, loop_config):
        FSDPDPOTrainer(loop_config).run()


class FSDPDPOTrainer:
    def __init__(self, loop_config):
        self.cfg: AppConfig = loop_config["config"]
        self.inference = loop_config["inference"]
        self.control = loop_config["control"]
        self.tracker = loop_config["tracker"]
        self.queue = loop_config["queue"]

        ctx = train.get_context()
        self.rank = ctx.get_world_rank()
        self.world = ctx.get_world_size()
        self.local_rank = ctx.get_local_rank()
        self.local_world = ctx.get_local_world_size()
        self.node_rank = ctx.get_node_rank()

        self.device = get_device()
        self.step = 0
        self.used_pairs = 0
        self.version = 0

        self.per_rank_batch_size = self.cfg.train.resolved_per_rank_batch_size()
        self.grad_accum_steps = self.cfg.train.gradient_accumulation_steps
        self.global_batch_size = self.cfg.train.global_batch_size

    def run(self):
        self._setup()

        while self.step < self.cfg.train.max_steps:
            if self._stop():
                break

            microbatches = self._next_batch()
            if not microbatches:
                time.sleep(self.cfg.train.poll_seconds)
                continue

            loss = self._step(microbatches)
            self.step += 1
            self.used_pairs += self.global_batch_size

            if self.rank == 0:
                ray.get(self.control.update.remote(
                    optimizer_step=self.step,
                    used_pairs=self.used_pairs,
                ))
                self.tracker.log.remote({
                    "axis/optimizer_step": self.step,
                    "axis/used_pairs": self.used_pairs,
                    "train/dpo_loss": loss,
                    "train/used_pairs": self.used_pairs,
                    "train/global_batch_size": self.global_batch_size,
                    "train/per_rank_batch_size": self.per_rank_batch_size,
                    "train/gradient_accumulation_steps": self.grad_accum_steps,
                    "train/world_size": self.world,
                    "train/local_world_size": self.local_world,
                })

            if self.step % self.cfg.train.sync_every == 0:
                self._sync_inference()

            train.report({
                "step": self.step,
                "dpo_loss": loss,
                "used_pairs": self.used_pairs,
                "global_batch_size": self.global_batch_size,
            })

        if self.rank == 0:
            self.control.request_stop.remote()

    def _setup(self):
        m = self.cfg.model
        dtype = torch.bfloat16

        self._validate_hybrid_topology()

        self.tok = AutoTokenizer.from_pretrained(
            m.policy_name,
            trust_remote_code=m.trust_remote_code,
        )
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        # Keep the full model on CPU before FSDP wrapping. FSDP's device_id then
        # moves/shards one FSDP unit at a time instead of materializing the entire
        # model on one GPU first.
        model = AutoModelForCausalLM.from_pretrained(
            m.policy_name,
            dtype=dtype,
            trust_remote_code=m.trust_remote_code,
        )
        model.config.use_cache = False

        if self.cfg.train.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if self.world > 1:
            auto_wrap_policy, layer_names = self._build_transformer_auto_wrap_policy(model)
            strategy = self._sharding_strategy()

            self.model = prepare_model(
                model,
                move_to_device=False,
                parallel_strategy="fsdp",
                parallel_strategy_kwargs={
                    "sharding_strategy": strategy,
                    "use_orig_params": True,
                    "auto_wrap_policy": auto_wrap_policy,
                    "device_id": self.device,
                    "limit_all_gathers": True,
                },
            )

            if self.rank == 0:
                print(
                    "FSDP setup:",
                    {
                        "strategy": self.cfg.train.sharding_strategy,
                        "world_size": self.world,
                        "local_world_size": self.local_world,
                        "transformer_layers": sorted(layer_names),
                        "global_batch_size": self.global_batch_size,
                        "per_rank_batch_size": self.per_rank_batch_size,
                        "gradient_accumulation_steps": self.grad_accum_steps,
                    },
                )
        else:
            # Ray prepare_model intentionally does not wrap FSDP at world_size=1.
            # Move the single-GPU smoke-test model normally.
            self.model = prepare_model(
                model,
                move_to_device=True,
                parallel_strategy=None,
            )
            if self.rank == 0:
                print(
                    "Single-rank training: FSDP/HYBRID_SHARD is inactive; "
                    "running ordinary full-parameter training on one GPU."
                )

        bad = [
            (name, p.dtype)
            for name, p in self.model.named_parameters()
            if p.is_floating_point() and p.dtype != torch.bfloat16
        ]
        if bad:
            raise RuntimeError(f"Policy contains non-BF16 floating parameters: {bad[:5]}")

        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.collate = DPOCollator(self.tok, self.cfg.train.max_length)

    def _validate_hybrid_topology(self):
        if self.world <= 1 or self.cfg.train.sharding_strategy != "hybrid_shard":
            return

        mine = {
            "rank": self.rank,
            "node_rank": self.node_rank,
            "local_world_size": self.local_world,
        }
        topology = [None for _ in range(self.world)]
        dist.all_gather_object(topology, mine)

        node_sizes = {}
        for item in topology:
            node_sizes[item["node_rank"]] = item["local_world_size"]

        # HYBRID_SHARD automatically forms an intra-node shard group and an
        # inter-node replica group. Equal train-worker counts per node make those
        # replica groups well-defined and keep the same shard index on each node.
        if len(set(node_sizes.values())) != 1:
            raise RuntimeError(
                "HYBRID_SHARD requires a balanced training topology: each node "
                f"must host the same number of training ranks. Got {node_sizes}. "
                "Reserve/pack training GPUs evenly per node."
            )

        if self.rank == 0:
            print("HYBRID_SHARD topology:", {"nodes": len(node_sizes), "train_ranks_per_node": node_sizes})

    def _build_transformer_auto_wrap_policy(self, model):
        # Hugging Face causal LMs normally expose transformer block class names in
        # _no_split_modules, e.g. Qwen2DecoderLayer / LlamaDecoderLayer.
        target_names = set(getattr(model, "_no_split_modules", []) or [])
        layer_classes = {
            type(module)
            for module in model.modules()
            if module.__class__.__name__ in target_names
        }

        # Conservative fallback for decoder-only LMs whose config does not expose
        # _no_split_modules.
        if not layer_classes:
            layer_classes = {
                type(module)
                for module in model.modules()
                if module.__class__.__name__.endswith("DecoderLayer")
            }

        if not layer_classes:
            raise RuntimeError(
                "Could not identify transformer block classes for FSDP auto-wrap. "
                "Expected Hugging Face _no_split_modules or *DecoderLayer modules."
            )

        policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=layer_classes,
        )
        return policy, {cls.__name__ for cls in layer_classes}

    def _sharding_strategy(self):
        if self.cfg.train.sharding_strategy == "hybrid_shard":
            return ShardingStrategy.HYBRID_SHARD
        if self.cfg.train.sharding_strategy == "full_shard":
            return ShardingStrategy.FULL_SHARD
        raise ValueError(f"Unknown sharding strategy: {self.cfg.train.sharding_strategy}")

    def _stop(self):
        flag = ray.get(self.control.should_stop.remote()) if self.rank == 0 else False
        obj = [flag]
        dist.broadcast_object_list(obj, src=0)
        return obj[0]

    def _next_batch(self):
        """
        Pop one EFFECTIVE global optimizer batch, then split it as:

          global_batch_size
            = per_rank_batch_size * world_size * gradient_accumulation_steps

        Each rank receives `gradient_accumulation_steps` microbatches, and every
        microbatch has exactly `per_rank_batch_size` preference pairs.
        """
        if self.rank == 0:
            items, stale = ray.get(self.queue.pop.remote(
                "preference",
                self.global_batch_size,
                self.version,
                self.cfg.train.max_policy_lag,
                True,
            ))
            if stale:
                self.tracker.log.remote({"queue/stale_dropped": stale})
            payload = items if len(items) == self.global_batch_size else None
        else:
            payload = None

        obj = [payload]
        dist.broadcast_object_list(obj, src=0)
        payload = obj[0]

        if not payload:
            return None

        local_total = self.per_rank_batch_size * self.grad_accum_steps
        start = self.rank * local_total
        end = start + local_total
        local_samples = payload[start:end]

        if len(local_samples) != local_total:
            raise RuntimeError(
                f"Rank {self.rank} expected {local_total} local samples, "
                f"got {len(local_samples)}"
            )

        return [
            local_samples[i:i + self.per_rank_batch_size]
            for i in range(0, local_total, self.per_rank_batch_size)
        ]

    def _step(self, microbatches):
        self.opt.zero_grad(set_to_none=True)
        total_loss = 0.0

        for micro_idx, samples in enumerate(microbatches):
            b = {
                k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in self.collate(samples).items()
            }

            # We deliberately keep gradient synchronization enabled on each
            # micro-step. FSDP no_sync() retains full gradients and can sharply
            # increase memory usage, which works against the purpose of FSDP.
            sync_ctx = nullcontext()
            with sync_ctx:
                out = self.model(
                    input_ids=b["input_ids"],
                    attention_mask=b["attention_mask"],
                )
                logps = self._sequence_logps(
                    out.logits,
                    b["input_ids"],
                    b["response_mask"],
                )
                n = b["pairs"]
                pi_c, pi_r = logps[:n], logps[n:]
                ref_c, ref_r = b["ref_logps"][:n], b["ref_logps"][n:]
                loss = -F.logsigmoid(
                    self.cfg.train.beta
                    * ((pi_c - pi_r) - (ref_c - ref_r))
                ).mean()

                (loss / self.grad_accum_steps).backward()

            total_loss += float(loss.detach())

        self.opt.step()
        return total_loss / len(microbatches)

    @staticmethod
    def _sequence_logps(logits, ids, response_mask):
        token_logps = (
            logits[:, :-1]
            .log_softmax(-1)
            .gather(-1, ids[:, 1:].unsqueeze(-1))
            .squeeze(-1)
        )
        return (token_logps * response_mask[:, 1:]).sum(-1)

    def _sync_inference(self):
        path = Path(self.cfg.train.checkpoint_dir) / f"policy_v{self.version + 1}"

        if isinstance(self.model, FSDP):
            state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                state_cfg,
            ):
                state = self.model.state_dict()
        else:
            # world_size=1 smoke test: Ray did not wrap the model with FSDP.
            state = {
                k: v.detach().cpu()
                for k, v in self.model.state_dict().items()
            }

        if self.rank == 0:
            path.mkdir(parents=True, exist_ok=True)
            torch.save(state, path / "pytorch_model.bin")
            base = self.model.module if hasattr(self.model, "module") else self.model
            base.config.save_pretrained(path)
            self.tok.save_pretrained(path)

        dist.barrier()

        if self.rank == 0:
            self.version += 1
            ray.get(self.inference.reload.remote(str(path), self.version))
            ray.get(self.control.update.remote(version=self.version))
            self.tracker.log.remote({
                "model/policy_version": self.version,
                "axis/optimizer_step": self.step,
                "axis/used_pairs": self.used_pairs,
            })

        obj = [self.version]
        dist.broadcast_object_list(obj, src=0)
        self.version = obj[0]
