"""Opt-in debugging helpers for the Ray/FSDP self-reward pipeline.

Nothing is printed or written unless --debug or a component-specific debug flag
is enabled.  Every Ray process writes to its own log file so interleaved worker
stdout does not hide where a hang happened.
"""

from __future__ import annotations

import faulthandler
import json
import os
import socket
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional


class Debugger:
    def __init__(self, config, component: str, rank: Optional[int] = None, stdout: Optional[bool] = None):
        self.config = config
        self.component = str(component).lower()
        self.rank = rank
        self.pid = os.getpid()
        self.host = socket.gethostname()
        self._log_path = None
        self._fault_file = None
        self._stage_stack = []
        self._stdout = bool(getattr(config, "worker_stdout", False)) if stdout is None else bool(stdout)

        if self.any_enabled():
            log_dir = Path(getattr(config, "log_dir", "./debug_logs"))
            log_dir.mkdir(parents=True, exist_ok=True)
            rank_part = f"-rank{rank}" if rank is not None else ""
            self._log_path = log_dir / f"{self.component}{rank_part}-pid{self.pid}.log"

    def any_enabled(self) -> bool:
        if self.config is None:
            return False
        if bool(getattr(self.config, "all", False)):
            return True
        return any(
            bool(getattr(self.config, name, False))
            for name in ("train", "fsdp", "queue", "ray", "reward", "inference", "data")
        )

    def enabled(self, category: Optional[str] = None) -> bool:
        if self.config is None:
            return False
        if bool(getattr(self.config, "all", False)):
            return True
        category = (category or self.component).lower()
        return bool(getattr(self.config, category, False))

    @staticmethod
    def _safe(value: Any):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [Debugger._safe(x) for x in value]
        if isinstance(value, dict):
            return {str(k): Debugger._safe(v) for k, v in value.items()}
        return repr(value)

    def log(self, category: str, event: str, **fields):
        if not self.enabled(category):
            return
        payload = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "category": category.upper(),
            "component": self.component,
            "event": event,
            "host": self.host,
            "pid": self.pid,
            "rank": self.rank,
            **{k: self._safe(v) for k, v in fields.items()},
        }
        line = "[DBG] " + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if self._stdout:
            print(line, flush=True)
        if self._log_path is not None:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def exception(self, category: str, event: str, exc: BaseException):
        if not self.enabled(category):
            return
        self.log(
            category,
            event,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )

    def _fault_stream(self):
        """Return a persistent stream for faulthandler.

        Worker/actor watchdog stacks go to a per-process file by default so they
        cannot overwrite HumanFeedback input. The main driver may opt into stderr.
        """
        if self._stdout:
            return sys.stderr
        if self._log_path is None:
            return sys.stderr
        if self._fault_file is None or self._fault_file.closed:
            stack_path = self._log_path.with_suffix(".stack.log")
            self._fault_file = stack_path.open("a", encoding="utf-8", buffering=1)
        return self._fault_file

    def _arm_watchdog(self, stage_name: str):
        stall_seconds = float(getattr(self.config, "stall_seconds", 0.0) or 0.0)
        if stall_seconds <= 0:
            return False
        try:
            stream = self._fault_stream()
            faulthandler.enable(file=stream)
            faulthandler.dump_traceback_later(stall_seconds, repeat=False, file=stream)
            self.log(self._stage_stack[-1][0], "watchdog_armed", stage=stage_name, timeout_sec=stall_seconds)
            return True
        except Exception as exc:
            self.log(self._stage_stack[-1][0], "watchdog_arm_failed", stage=stage_name, error=repr(exc))
            return False

    @staticmethod
    def _cancel_watchdog():
        try:
            faulthandler.cancel_dump_traceback_later()
        except Exception:
            pass

    @contextmanager
    def stage(self, category: str, name: str, **fields):
        """Print BEGIN/END and dump a Python stack if this stage stalls.

        If BEGIN appears without END, the stuck operation is the stage named in
        that BEGIN record. Nested stages are supported: after a child stage ends,
        the watchdog is re-armed for its parent stage.
        """
        if not self.enabled(category):
            yield
            return

        started = time.monotonic()
        self.log(category, f"BEGIN:{name}", **fields)
        self._stage_stack.append((category, name, started))
        self._cancel_watchdog()
        self._arm_watchdog(name)

        try:
            yield
        except BaseException as exc:
            self.exception(category, f"ERROR:{name}", exc)
            raise
        finally:
            self._cancel_watchdog()
            elapsed = round(time.monotonic() - started, 6)
            if self._stage_stack:
                self._stage_stack.pop()
            self.log(category, f"END:{name}", elapsed_sec=elapsed)
            if self._stage_stack:
                parent_category, parent_name, _ = self._stage_stack[-1]
                self._arm_watchdog(parent_name)

    def close(self):
        if self._fault_file is not None and not self._fault_file.closed:
            try:
                self._fault_file.close()
            except Exception:
                pass

    def __del__(self):
        self.close()

    def environment(self, category: str, **extra):
        if not self.enabled(category):
            return
        info = {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "RANK": os.environ.get("RANK"),
            "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
            "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
            "MASTER_ADDR": os.environ.get("MASTER_ADDR"),
            "MASTER_PORT": os.environ.get("MASTER_PORT"),
            **extra,
        }
        try:
            import torch
            info.update({
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            })
        except Exception as exc:
            info["torch_probe_error"] = repr(exc)
        self.log(category, "environment", **info)
