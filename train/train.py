import os
import time
from pathlib import Path

import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
    ShardingStrategy,
)
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
        ctx = train.get_context(); self.rank, self.world = ctx.get_world_rank(), ctx.get_world_size()
        self.device = get_device(); self.step = self.used_pairs = self.version = 0

    def run(self):
        self._setup()
        while self.step < self.cfg.train.max_steps:
            if self._stop(): break
            batch = self._next_batch()
            if not batch: time.sleep(self.cfg.train.poll_seconds); continue
            loss = self._step(batch)
            self.step += 1; self.used_pairs += len(batch)
            if self.rank == 0:
                ray.get(self.control.update.remote(optimizer_step=self.step, used_pairs=self.used_pairs))
                self.tracker.log.remote({"axis/optimizer_step": self.step, "axis/used_pairs": self.used_pairs,
                                         "train/dpo_loss": loss, "train/used_pairs": self.used_pairs})
            if self.step % self.cfg.train.sync_every == 0: self._sync_inference()
            train.report({"step": self.step, "dpo_loss": loss})
        if self.rank == 0:
            self.control.request_stop.remote()

    def _setup(self):
        m = self.cfg.model
        dtype = torch.bfloat16
        self.tok = AutoTokenizer.from_pretrained(m.policy_name, trust_remote_code=m.trust_remote_code)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(m.policy_name, dtype=dtype,
                                                     trust_remote_code=m.trust_remote_code)
        model.config.use_cache = False
        if self.cfg.train.gradient_checkpointing: model.gradient_checkpointing_enable()
        self.model = prepare_model(model, parallel_strategy="fsdp", parallel_strategy_kwargs={
            "sharding_strategy": ShardingStrategy.FULL_SHARD, "use_orig_params": True})

        bad = [
            (name, p.dtype)
            for name, p in self.model.named_parameters()
            if p.is_floating_point() and p.dtype != torch.bfloat16
        ]
        if bad:
            raise RuntimeError(f"Policy contains non-BF16 floating parameters: {bad[:5]}")

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.train.learning_rate,
                                     weight_decay=self.cfg.train.weight_decay)
        self.collate = DPOCollator(self.tok, self.cfg.train.max_length)

    def _stop(self):
        flag = ray.get(self.control.should_stop.remote()) if self.rank == 0 else False
        obj = [flag]; dist.broadcast_object_list(obj, src=0)
        return obj[0]

    def _next_batch(self):
        if self.rank == 0:
            items, stale = ray.get(self.queue.pop.remote(
                "preference",
                self.cfg.train.batch_size,
                self.version,
                self.cfg.train.max_policy_lag,
                True,
            ))
            if stale: self.tracker.log.remote({"queue/stale_dropped": stale})
            payload = items if len(items) == self.cfg.train.batch_size else None
        else: payload = None
        obj = [payload]; dist.broadcast_object_list(obj, src=0)
        if not obj[0]: return None
        return obj[0][self.rank::self.world]

    def _step(self, samples):
        b = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in self.collate(samples).items()}
        self.opt.zero_grad(); out = self.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        logps = self._sequence_logps(out.logits, b["input_ids"], b["response_mask"])
        n = b["pairs"]; pi_c, pi_r = logps[:n], logps[n:]
        ref_c, ref_r = b["ref_logps"][:n], b["ref_logps"][n:]
        loss = -F.logsigmoid(self.cfg.train.beta * ((pi_c - pi_r) - (ref_c - ref_r))).mean()
        loss.backward(); self.opt.step()
        return float(loss.detach())

    @staticmethod
    def _sequence_logps(logits, ids, response_mask):
        token_logps = logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        return (token_logps * response_mask[:, 1:]).sum(-1)

    def _sync_inference(self):
        path = Path(self.cfg.train.checkpoint_dir) / f"policy_v{self.version + 1}"
        state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, state_cfg):
            state = self.model.state_dict()
        if self.rank == 0:
            path.mkdir(parents=True, exist_ok=True)
            torch.save(state, path / "pytorch_model.bin")
            base = self.model.module if hasattr(self.model, "module") else self.model
            base.config.save_pretrained(path); self.tok.save_pretrained(path)
        dist.barrier()
        if self.rank == 0:
            self.version += 1
            ray.get(self.inference.reload.remote(str(path), self.version))
            ray.get(self.control.update.remote(version=self.version))
            self.tracker.log.remote({"model/policy_version": self.version, "axis/optimizer_step": self.step,
                                     "axis/used_pairs": self.used_pairs})
        obj = [self.version]; dist.broadcast_object_list(obj, src=0); self.version = obj[0]
