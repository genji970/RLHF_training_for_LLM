import json
import os
import time
from pathlib import Path


class MetricTracker:
    """Single logging sink for JSONL + W&B. Make this a Ray actor and share the handle."""
    def __init__(self, config):
        self.dir = Path(config.output_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.file = open(self.dir / "metrics.jsonl", "a", buffering=1)
        self.wandb = None
        if config.wandb:
            import wandb
            self.wandb = wandb
            wandb.init(project=config.project, name=config.run_name, dir=str(self.dir))
            for x in ("axis/optimizer_step", "axis/human_groups", "axis/used_pairs"): wandb.define_metric(x)
            wandb.define_metric("eval/by_optimizer_step", step_metric="axis/optimizer_step")
            wandb.define_metric("eval/by_human_groups", step_metric="axis/human_groups")
            wandb.define_metric("eval/by_used_pairs", step_metric="axis/used_pairs")

    def log(self, metrics):
        row = {"time": time.time(), **metrics}
        self.file.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.wandb: self.wandb.log(metrics)

    def labeled(self, item):
        with open(self.dir / "feedback.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def close(self):
        self.file.close()
        if self.wandb: self.wandb.finish()


class TrainControl:
    def __init__(self):
        self.stop = False
        self.version = 0
        self.optimizer_step = 0
        self.used_pairs = 0

    def should_stop(self): return self.stop
    def request_stop(self): self.stop = True
    def update(self, version=None, optimizer_step=None, used_pairs=None):
        if version is not None: self.version = int(version)
        if optimizer_step is not None: self.optimizer_step = int(optimizer_step)
        if used_pairs is not None: self.used_pairs = int(used_pairs)
    def status(self):
        return {"version": self.version, "optimizer_step": self.optimizer_step, "used_pairs": self.used_pairs}
