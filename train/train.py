import functools
import json
import time
import traceback
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import AppConfig
from data import DPOCollator
from debug import Debugger


class DistributedDPOTrainLoop:
    def __call__(self, loop_config):
        cfg = loop_config["config"]
        ctx = train.get_context()
        rank = ctx.get_world_rank()
        dbg = Debugger(cfg.debug, "train", rank=rank)
        dbg.log("train", "worker_entry")
        try:
            FSDPDPOTrainer(loop_config, dbg).run()
        except BaseException as exc:
            dbg.exception("train", "worker_crashed", exc)
            if dbg.enabled("train") or dbg.enabled("fsdp"):
                print(f"[TRAIN][rank={rank}] worker crashed", flush=True)
                traceback.print_exc()
            raise


class FSDPDPOTrainer:
    def __init__(self, loop_config, debugger=None):
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

        # Ray Train may expose multiple local CUDA devices to every worker. NCCL
        # object collectives use torch.cuda.current_device(), so bind the process
        # explicitly to the device assigned by Ray before any collective runs.
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        self.debug = debugger or Debugger(self.cfg.debug, "train", rank=self.rank)
        self.debug.environment(
            "train",
            world=self.world,
            local_rank=self.local_rank,
            local_world=self.local_world,
            node_rank=self.node_rank,
            device=str(self.device),
        )

        self.step = 0
        self.used_pairs = 0
        self.version = 0
        self.per_rank_batch_size = self.cfg.train.resolved_per_rank_batch_size()
        self.grad_accum_steps = self.cfg.train.gradient_accumulation_steps
        self.global_batch_size = self.cfg.train.global_batch_size
        self._empty_polls = 0

        self.debug.log(
            "train",
            "trainer_init",
            global_batch_size=self.global_batch_size,
            per_rank_batch_size=self.per_rank_batch_size,
            gradient_accumulation_steps=self.grad_accum_steps,
            sharding_strategy=self.cfg.train.sharding_strategy,
        )

    def run(self):
        with self.debug.stage("train", "setup"):
            self._setup()

        self.debug.log("train", "train_loop_ready")
        while self.step < self.cfg.train.max_steps:
            if self._stop():
                self.debug.log("train", "stop_requested", step=self.step)
                break

            microbatches = self._next_batch()
            if not microbatches:
                time.sleep(self.cfg.train.poll_seconds)
                continue

            with self.debug.stage("train", "optimizer_step", next_step=self.step + 1):
                loss = self._step(microbatches)

            self.step += 1
            self.used_pairs += self.global_batch_size

            if self.rank == 0:
                with self.debug.stage("ray", "control_update", step=self.step):
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
                self.debug.log(
                    "train",
                    "step_done",
                    step=self.step,
                    loss=loss,
                    used_pairs=self.used_pairs,
                    policy_version=self.version,
                )

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
        self.debug.log("train", "train_loop_exit", step=self.step, used_pairs=self.used_pairs)

    def _setup(self):
        m = self.cfg.model
        dtype = torch.bfloat16

        # Keep this check instrumented instead of silently removing it. If this
        # collective is the hang, debug output will show BEGIN without END.
        with self.debug.stage(
            "fsdp",
            "hybrid_topology_check",
            world=self.world,
            local_world=self.local_world,
            node_rank=self.node_rank,
        ):
            self._validate_hybrid_topology()

        # Keep trainer control-plane collectives off the NCCL/CUDA stream.
        # Ray/FSDP owns the default NCCL group for model collectives; a separate
        # Gloo group is used for tiny stop flags, queue payloads, and versions.
        if self.world > 1:
            with self.debug.stage("train", "control_group_create", backend="gloo"):
                self.control_group = dist.new_group(backend="gloo")
        else:
            self.control_group = None

        with self.debug.stage("train", "tokenizer_load", model=m.policy_name):
            self.tok = AutoTokenizer.from_pretrained(
                m.policy_name,
                trust_remote_code=m.trust_remote_code,
            )
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        # Load on CPU first; FSDP's device_id moves/shards wrapped units.
        with self.debug.stage("train", "model_load_cpu", model=m.policy_name, dtype=str(dtype)):
            model = AutoModelForCausalLM.from_pretrained(
                m.policy_name,
                dtype=dtype,
                trust_remote_code=m.trust_remote_code,
            )
        model.config.use_cache = False

        if self.cfg.train.gradient_checkpointing:
            with self.debug.stage("train", "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

        if self.world > 1:
            with self.debug.stage("fsdp", "find_transformer_layers"):
                auto_wrap_policy, layer_names = self._build_transformer_auto_wrap_policy(model)
            strategy = self._sharding_strategy()
            self.debug.log(
                "fsdp",
                "wrap_config",
                strategy=str(strategy),
                transformer_layers=sorted(layer_names),
                device=str(self.device),
            )

            with self.debug.stage(
                "fsdp",
                "fsdp_wrap",
                strategy=self.cfg.train.sharding_strategy,
                device=str(self.device),
            ):
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
        else:
            with self.debug.stage("train", "single_gpu_prepare_model", device=str(self.device)):
                self.model = prepare_model(
                    model,
                    move_to_device=True,
                    parallel_strategy=None,
                )
            self.debug.log("fsdp", "fsdp_inactive_world_size_1")

        with self.debug.stage("train", "bf16_parameter_check"):
            bad = [
                (name, p.dtype)
                for name, p in self.model.named_parameters()
                if p.is_floating_point() and p.dtype != torch.bfloat16
            ]
            if bad:
                raise RuntimeError(f"Policy contains non-BF16 floating parameters: {bad[:5]}")

        with self.debug.stage("train", "optimizer_create"):
            self.opt = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.train.learning_rate,
                weight_decay=self.cfg.train.weight_decay,
            )
        self.collate = DPOCollator(self.tok, self.cfg.train.max_length)
        self.debug.log("train", "setup_done", model_type=type(self.model).__name__)

    def _validate_hybrid_topology(self):
        if self.world <= 1 or self.cfg.train.sharding_strategy != "hybrid_shard":
            return

        # Do not run a Python-object collective here. With NCCL,
        # all_gather_object()/broadcast_object_list() use the process current CUDA
        # device internally and can deadlock before FSDP setup if the worker device
        # has not been bound yet. Ray/FSDP already owns process-group construction.
        # Keep this as a local sanity check only; FSDP will validate the actual
        # sharding/replication process groups during wrapping.
        if self.local_world < 1 or self.world < self.local_world:
            raise RuntimeError(
                f"Invalid Ray Train topology: world={self.world}, "
                f"local_world={self.local_world}, node_rank={self.node_rank}."
            )

        self.debug.log(
            "fsdp",
            "topology_local_check",
            world=self.world,
            local_world=self.local_world,
            node_rank=self.node_rank,
            current_cuda_device=(
                torch.cuda.current_device() if torch.cuda.is_available() else None
            ),
            note="collective topology probe intentionally skipped; FSDP owns group setup",
        )

    @staticmethod
    def _build_transformer_auto_wrap_policy(model):
        target_names = set(getattr(model, "_no_split_modules", []) or [])
        layer_classes = {
            type(module)
            for module in model.modules()
            if module.__class__.__name__ in target_names
        }
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
        requested = self.cfg.train.sharding_strategy

        if requested == "hybrid_shard":
            # HYBRID_SHARD means FULL_SHARD inside each node plus replication
            # across nodes. On a single-node job there is no replication axis,
            # so resolve it to ordinary FULL_SHARD. This avoids constructing a
            # degenerate inter-node replication group and keeps the exact same
            # semantics for the one-node case.
            if self.world == self.local_world:
                self.debug.log(
                    "fsdp",
                    "sharding_strategy_resolved",
                    requested=requested,
                    resolved="full_shard",
                    reason="single_node_world_equals_local_world",
                    world=self.world,
                    local_world=self.local_world,
                )
                return ShardingStrategy.FULL_SHARD

            self.debug.log(
                "fsdp",
                "sharding_strategy_resolved",
                requested=requested,
                resolved="hybrid_shard",
                reason="multi_node",
                world=self.world,
                local_world=self.local_world,
            )
            return ShardingStrategy.HYBRID_SHARD

        if requested == "full_shard":
            self.debug.log(
                "fsdp",
                "sharding_strategy_resolved",
                requested=requested,
                resolved="full_shard",
                reason="explicit",
                world=self.world,
                local_world=self.local_world,
            )
            return ShardingStrategy.FULL_SHARD

        raise ValueError(f"Unknown sharding strategy: {requested}")

    def _stop(self):
        if self.rank == 0:
            with self.debug.stage("ray", "control_should_stop"):
                flag = bool(ray.get(self.control.should_stop.remote()))
        else:
            flag = False

        # CPU/Gloo control-plane broadcast. This intentionally avoids CUDA
        # synchronization from flag_tensor.item() on NCCL tensors.
        flag_tensor = torch.tensor(
            [1 if flag else 0],
            dtype=torch.uint8,
            device="cpu",
        )
        with self.debug.stage(
            "train",
            "broadcast_stop_flag",
            value=flag if self.rank == 0 else None,
            transport="gloo_cpu",
        ):
            if self.world > 1:
                dist.broadcast(flag_tensor, src=0, group=self.control_group)
            result = bool(flag_tensor.item())
        return result

    def _broadcast_json_payload(self, payload):
        """Broadcast a JSON-compatible payload over the CPU/Gloo control group."""
        if self.rank == 0 and payload is not None:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            byte_count = len(encoded)
        else:
            encoded = None
            byte_count = 0

        length = torch.tensor([byte_count], dtype=torch.int64, device="cpu")
        with self.debug.stage(
            "train",
            "broadcast_preference_length",
            bytes=byte_count if self.rank == 0 else None,
            transport="gloo_cpu",
        ):
            if self.world > 1:
                dist.broadcast(length, src=0, group=self.control_group)
            byte_count = int(length.item())

        if byte_count == 0:
            return None

        if self.rank == 0:
            data = torch.tensor(list(encoded), dtype=torch.uint8, device="cpu")
        else:
            data = torch.empty(byte_count, dtype=torch.uint8, device="cpu")

        with self.debug.stage(
            "train",
            "broadcast_preference_bytes",
            bytes=byte_count,
            transport="gloo_cpu",
        ):
            if self.world > 1:
                dist.broadcast(data, src=0, group=self.control_group)

        if self.rank == 0:
            return payload
        return json.loads(bytes(data.tolist()).decode("utf-8"))

    def _next_batch(self):
        if self.rank == 0:
            with self.debug.stage("queue", "preference_size_before_pop"):
                queue_size = ray.get(self.queue.size.remote("preference"))
            self.debug.log(
                "queue",
                "trainer_queue_state",
                pending=queue_size,
                required=self.global_batch_size,
                policy_version=self.version,
            )

            with self.debug.stage(
                "queue",
                "preference_pop",
                requested=self.global_batch_size,
                current_version=self.version,
            ):
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
            if payload is None:
                self._empty_polls += 1
                if self._empty_polls == 1 or self._empty_polls % 20 == 0:
                    self.debug.log(
                        "train",
                        "waiting_for_preference_batch",
                        pending=queue_size,
                        required=self.global_batch_size,
                        empty_polls=self._empty_polls,
                    )
            else:
                self._empty_polls = 0
                self.debug.log("queue", "trainer_pop_done", count=len(items))
        else:
            payload = None

        with self.debug.stage(
            "train",
            "broadcast_preference_payload",
            has_payload=payload is not None if self.rank == 0 else None,
            transport="json_uint8_gloo",
        ):
            payload = self._broadcast_json_payload(payload)
        if not payload:
            return None

        local_total = self.per_rank_batch_size * self.grad_accum_steps
        start = self.rank * local_total
        end = start + local_total
        local_samples = payload[start:end]
        if len(local_samples) != local_total:
            raise RuntimeError(
                f"Rank {self.rank} expected {local_total} local samples, got {len(local_samples)}"
            )

        microbatches = [
            local_samples[i:i + self.per_rank_batch_size]
            for i in range(0, local_total, self.per_rank_batch_size)
        ]
        self.debug.log(
            "train",
            "local_batch_ready",
            local_total=local_total,
            microbatches=len(microbatches),
            microbatch_size=self.per_rank_batch_size,
        )
        return microbatches

    def _step(self, microbatches):
        with self.debug.stage("train", "zero_grad"):
            self.opt.zero_grad(set_to_none=True)
        total_loss = 0.0

        for micro_idx, samples in enumerate(microbatches):
            with self.debug.stage("train", "collate", micro_idx=micro_idx, pairs=len(samples)):
                raw_batch = self.collate(samples)
            with self.debug.stage("train", "batch_to_device", micro_idx=micro_idx, device=str(self.device)):
                b = {
                    k: (v.to(self.device) if torch.is_tensor(v) else v)
                    for k, v in raw_batch.items()
                }

            self.debug.log(
                "train",
                "batch_shapes",
                micro_idx=micro_idx,
                input_ids=tuple(b["input_ids"].shape),
                attention_mask=tuple(b["attention_mask"].shape),
                response_mask=tuple(b["response_mask"].shape),
            )

            with self.debug.stage("fsdp", "forward", micro_idx=micro_idx):
                out = self.model(
                    input_ids=b["input_ids"],
                    attention_mask=b["attention_mask"],
                )

            with self.debug.stage("train", "dpo_loss", micro_idx=micro_idx):
                logps = self._sequence_logps(
                    out.logits,
                    b["input_ids"],
                    b["response_mask"],
                )
                n = b["pairs"]
                pi_c, pi_r = logps[:n], logps[n:]
                ref_c, ref_r = b["ref_logps"][:n], b["ref_logps"][n:]
                loss = -F.logsigmoid(
                    self.cfg.train.beta * ((pi_c - pi_r) - (ref_c - ref_r))
                ).mean()

            with self.debug.stage("fsdp", "backward", micro_idx=micro_idx, loss=float(loss.detach())):
                (loss / self.grad_accum_steps).backward()
            total_loss += float(loss.detach())

        with self.debug.stage("train", "optimizer_apply"):
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
        checkpoint_root = Path(self.cfg.train.checkpoint_dir).expanduser()
        if not checkpoint_root.is_absolute():
            checkpoint_root = (Path.cwd() / checkpoint_root).resolve()
        else:
            checkpoint_root = checkpoint_root.resolve()
        path = checkpoint_root / f"policy_v{self.version + 1}"

        with self.debug.stage("fsdp", "full_state_dict", path=str(path)):
            if isinstance(self.model, FSDP):
                state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(
                    self.model,
                    StateDictType.FULL_STATE_DICT,
                    state_cfg,
                ):
                    state = self.model.state_dict()
            else:
                state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

        if self.rank == 0:
            with self.debug.stage("train", "checkpoint_save", path=str(path)):
                path.mkdir(parents=True, exist_ok=True)
                torch.save(state, path / "pytorch_model.bin")
                base = self.model.module if hasattr(self.model, "module") else self.model
                base.config.save_pretrained(path)
                self.tok.save_pretrained(path)

        with self.debug.stage("fsdp", "checkpoint_barrier"):
            dist.barrier()

        if self.rank == 0:
            if not path.is_dir():
                raise FileNotFoundError(f"Checkpoint directory does not exist after save: {path}")
            weight_file = path / "pytorch_model.bin"
            if not weight_file.is_file():
                raise FileNotFoundError(f"Checkpoint weight file is missing: {weight_file}")

            next_version = self.version + 1
            with self.debug.stage("inference", "vllm_reload", path=str(path), version=next_version):
                ray.get(self.inference.reload.remote(str(path), next_version))
            self.version = next_version
            ray.get(self.control.update.remote(version=self.version))
            self.tracker.log.remote({
                "model/policy_version": self.version,
                "axis/optimizer_step": self.step,
                "axis/used_pairs": self.used_pairs,
            })

        version_tensor = torch.tensor(
            [self.version if self.rank == 0 else 0],
            dtype=torch.int64,
            device="cpu",
        )
        with self.debug.stage(
            "train",
            "broadcast_policy_version",
            version=self.version if self.rank == 0 else None,
            transport="gloo_cpu",
        ):
            if self.world > 1:
                dist.broadcast(version_tensor, src=0, group=self.control_group)
            self.version = int(version_tensor.item())
        self.debug.log("train", "sync_done", policy_version=self.version)
