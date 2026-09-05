"""Run locks, detached process controls, and safe graceful-stop signaling."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .checkpointing import atomic_json


def process_alive(pid: int) -> bool:
    if pid <= 0: return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


class RunLock:
    def __init__(self, run_dir: Path, config_digest: str):
        self.run_dir = run_dir; self.path = run_dir / "run.lock"; self.config_digest = config_digest

    def acquire(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            current = json.loads(self.path.read_text(encoding="utf-8"))
            if process_alive(int(current.get("pid", -1))):
                raise ValueError(f"RUN_ALREADY_ACTIVE:{current['pid']}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(self.path, flags)
        except FileExistsError:
            self.path.unlink(missing_ok=True); descriptor = os.open(self.path, flags)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump({"pid": os.getpid(), "config_digest": self.config_digest, "started_at": time.time()}, handle)

    def release(self) -> None:
        if self.path.exists():
            current = json.loads(self.path.read_text(encoding="utf-8"))
            if int(current.get("pid", -1)) == os.getpid(): self.path.unlink()


def request_stop(run_dir: Path) -> dict[str, Any]:
    lock = run_dir / "run.lock"
    if not lock.is_file(): return {"status": "NOT_RUNNING"}
    identity = json.loads(lock.read_text(encoding="utf-8")); pid = int(identity["pid"])
    if not process_alive(pid): return {"status": "STALE_LOCK", "pid": pid}
    (run_dir / "STOP_REQUESTED").write_text(f"requested_at={time.time()}\n", encoding="utf-8", newline="\n")
    return {"status": "GRACEFUL_STOP_REQUESTED", "pid": pid}


def status(run_dir: Path) -> dict[str, Any]:
    value: dict[str, Any] = {"run_dir": str(run_dir), "active": False, "stop_requested": (run_dir / "STOP_REQUESTED").exists()}
    lock = run_dir / "run.lock"
    if lock.exists():
        identity = json.loads(lock.read_text(encoding="utf-8")); value.update(identity); value["active"] = process_alive(int(identity["pid"]))
    state = run_dir / "status.json"
    if state.exists(): value["training"] = json.loads(state.read_text(encoding="utf-8"))
    return value
