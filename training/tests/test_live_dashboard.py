from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from urllib.request import urlopen

import pytest
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from dataset_forge_critic.live_dashboard import build_snapshot, create_server


def _events(run_dir: Path, points: list[tuple[int, float, float, float]]) -> None:
    writer = EventFileWriter(str(run_dir / "tensorboard"))
    for step, wall_time, loss, learning_rate in points:
        writer.add_event(Event(wall_time=wall_time, step=step, summary=Summary(value=[
            Summary.Value(tag="train/loss", simple_value=loss),
            Summary.Value(tag="train/learning_rate", simple_value=learning_rate),
        ])))
    writer.flush()
    writer.close()


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _inventory(root: Path) -> dict[str, tuple[bytes, int]]:
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns) for path in root.rglob("*") if path.is_file()}


def test_progress_rate_eta_and_read_only(tmp_path: Path) -> None:
    start = 1_800_000_000.0
    _events(tmp_path, [(step, start + step * 60, 2.0 / step, 1e-4 / step) for step in range(1, 11)])
    status = tmp_path / "status.json"
    _write(status, {"step": 10, "elapsed_seconds": 600, "loss": 0.2, "learning_rate": 1e-5})
    os.utime(status, (start + 600, start + 600))
    before = _inventory(tmp_path)
    snapshot = build_snapshot(tmp_path, 20, now=start + 600)
    assert snapshot["current_step"] == 10
    assert snapshot["remaining_steps"] == 10
    assert snapshot["completion_percent"] == 50.0
    assert snapshot["steps_per_minute"] == pytest.approx(1.0)
    assert snapshot["eta_seconds"] == pytest.approx(600.0)
    assert snapshot["smoothed_loss"] is not None
    assert _inventory(tmp_path) == before


def test_completed_and_missing_metrics(tmp_path: Path) -> None:
    completed = tmp_path / "completed"; completed.mkdir()
    _write(completed / "result.json", {"status": "TRAINING_BUDGET_COMPLETE", "step": 20})
    done = build_snapshot(completed, 20)
    assert done["process_status"] == "completed" and done["remaining_steps"] == 0 and done["eta_seconds"] == 0
    missing = tmp_path / "missing"; missing.mkdir()
    empty = build_snapshot(missing, 20)
    assert empty["current_loss"] is None and empty["learning_rate"] is None and empty["eta_seconds"] is None


def test_http_server_starts_on_loopback(tmp_path: Path) -> None:
    _write(tmp_path / "status.json", {"step": 3})
    server = create_server(tmp_path, "127.0.0.1", 0, 10)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/api/status", timeout=3) as response:
            payload = json.load(response)
        assert payload["current_step"] == 3 and payload["total_steps"] == 10
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=3)
