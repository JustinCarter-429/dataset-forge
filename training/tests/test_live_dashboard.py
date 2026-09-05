from __future__ import annotations

import json, os, threading
from pathlib import Path
from urllib.request import urlopen

import pytest
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

import dataset_forge_critic.live_dashboard as dashboard


def _events(run_dir: Path, points: list[tuple[int, float, float, float]]) -> None:
    writer = EventFileWriter(str(run_dir / "tensorboard"))
    for step, wall_time, loss, learning_rate in points:
        writer.add_event(Event(wall_time=wall_time, step=step, summary=Summary(value=[
            Summary.Value(tag="train/loss", simple_value=loss),
            Summary.Value(tag="train/learning_rate", simple_value=learning_rate),
            Summary.Value(tag="throughput/supervised_tokens_per_second", simple_value=1200 + step),
        ])))
    writer.flush(); writer.close()


def _write(path: Path, value: dict) -> None: path.write_text(json.dumps(value), encoding="utf-8")


def _inventory(root: Path) -> dict[str, tuple[bytes, int]]:
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns) for path in root.rglob("*") if path.is_file()}


def test_mapping_progress_rolling_eta_smoothing_and_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    start = 1_800_000_000.0
    _events(tmp_path, [(step, start + step * 12, 1.0 / step, 1e-4 - step * 1e-8) for step in range(1, 101)])
    _write(tmp_path / "status.json", {"step": 100, "elapsed_seconds": 1200, "loss": 0.00058, "learning_rate": 9.9e-5})
    _write(tmp_path / "run.lock", {"pid": 3875, "started_at": start})
    os.utime(tmp_path / "status.json", (start + 1200, start + 1200)); monkeypatch.setattr(dashboard, "_process_alive", lambda pid: pid == 3875)
    before = _inventory(tmp_path); snap = dashboard.build_snapshot(tmp_path, 200, now=start + 1200)
    assert (snap["current_step"], snap["remaining_steps"], snap["completion_percent"]) == (100, 100, 50.0)
    assert snap["current_loss"] == pytest.approx(0.01)  # train/loss, not status LR-like value
    assert snap["learning_rate"] == pytest.approx(9.9e-5)
    assert snap["smoothed_loss"] == pytest.approx(sum(1.0 / s for s in range(81, 101)) / 20)
    assert snap["steps_per_minute"] == pytest.approx(5.0) and snap["eta_seconds"] == pytest.approx(1200)
    assert snap["process_status"] == "running" and snap["data_status"] == "Metrics updating"
    assert snap["metric_sources"]["Current loss"] == "TensorBoard scalar train/loss"
    assert snap["metric_sources"]["Learning rate"] == "TensorBoard scalar train/learning_rate"
    assert snap["metric_sources"]["Current step"].startswith("Maximum of status.json")
    assert snap["charts"]["supervised_throughput"] and _inventory(tmp_path) == before


def test_stale_running_completed_and_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    start = 1_800_000_000.0; running = tmp_path / "running"; running.mkdir()
    _events(running, [(s, start + s * 10, 1 / s, 1e-4) for s in range(1, 61)]); _write(running / "run.lock", {"pid": 3875, "started_at": start})
    monkeypatch.setattr(dashboard, "_process_alive", lambda pid: True)
    stale = dashboard.build_snapshot(running, 100, now=start + 2000)
    assert stale["process_status"] == "running" and stale["data_status"] == "Awaiting new metrics"
    _write(running / "status.json", {"step": 60, "elapsed_seconds": 2000})
    os.utime(running / "status.json", (start + 1995, start + 1995))
    active = dashboard.build_snapshot(running, 100, now=start + 2000)
    assert active["data_status"] == "Metrics updating"  # Status can lead a delayed event file.
    completed = tmp_path / "completed"; completed.mkdir(); _write(completed / "result.json", {"status": "TRAINING_BUDGET_COMPLETE", "step": 20})
    done = dashboard.build_snapshot(completed, 20)
    assert done["process_status"] == "completed" and done["remaining_steps"] == 0 and done["eta_seconds"] == 0
    missing = tmp_path / "missing"; missing.mkdir(); empty = dashboard.build_snapshot(missing, 20)
    assert empty["current_loss"] is None and empty["learning_rate"] is None and empty["eta_seconds"] is None and empty["missing_metrics"]


def test_active_run_isolation_html_escaping_and_http(tmp_path: Path) -> None:
    active = tmp_path / "full-active"; active.mkdir(parents=True)
    smoke = tmp_path / "smoke"; smoke.mkdir(); _events(active, [(7, 1_800_000_000, .7, 7e-5)]); _events(smoke, [(99, 1_800_000_000, .01, 1e-6)])
    _write(active / "run-manifest.json", {"model_id": "<img onerror=alert(1)>"})
    server = dashboard.create_server(active, "127.0.0.1", 0, 100)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/api/status", timeout=3) as response: payload=json.load(response)
        with urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=3) as response: html=response.read().decode()
        with urlopen(f"http://127.0.0.1:{server.server_port}/healthz", timeout=3) as response: health=json.load(response)
        assert payload["current_step"] == 7 and payload["run_name"].startswith("full-")
        assert payload["model"] == "<img onerror=alert(1)>"
        assert "<img onerror=alert(1)>" not in html and "textContent" in html
        assert health == {"status": "ok"}
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=3)


def test_checkpoint_timestamp_step_and_malformed_metadata_warning(tmp_path: Path) -> None:
    _write(tmp_path / "latest-checkpoint.json", {
        "path": "/workspace/run/checkpoints/step-00000500",
        "step": 500,
        "timestamp": "2027-01-15T12:00:00Z",
    })
    (tmp_path / "run-manifest.json").write_text("{broken", encoding="utf-8")
    snap = dashboard.build_snapshot(tmp_path, 1000, now=1_800_000_000)
    assert snap["latest_checkpoint_name"] == "step-00000500"
    assert snap["latest_checkpoint_step"] == 500
    assert snap["checkpoint_timestamp"] is not None
    assert any("run-manifest.json" in warning for warning in snap["warnings"])


def test_chart_math_trend_downsampling_and_server_guards(tmp_path: Path) -> None:
    assert dashboard._moving_average([1.0, 2.0, 3.0], 2) == [1.0, 1.5, 2.5]
    assert dashboard._trend([2.0 - i * .02 for i in range(60)]) == "Improving"
    points = [{"step": i, "value": 100.0 if i == 500 else 1.0, "wall_time": float(i)} for i in range(1000)]
    sampled = dashboard._downsample(points, limit=100)
    assert any(point["step"] == 500 and point["value"] == 100.0 for point in sampled)
    with pytest.raises(ValueError, match="LIVE_DASHBOARD_LOOPBACK_ONLY"):
        dashboard.create_server(tmp_path, "0.0.0.0", 0, 100)
    with pytest.raises(ValueError, match="TOTAL_STEPS_MUST_BE_POSITIVE"):
        dashboard.create_server(tmp_path, "127.0.0.1", 0, 0)


def test_process_probe_treats_platform_probe_errors_as_not_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_probe(pid: int, signal: int) -> None:
        raise SystemError("platform process probe failed")
    monkeypatch.setattr(dashboard.os, "kill", fail_probe)
    assert dashboard._process_alive(3875) is False


def test_html_contains_monitoring_interactions_and_external_chart_pin() -> None:
    for required in (
        "Monitoring only — training controls are disabled", "Auto refresh", "Pause charts",
        "Last 100", "Last 500", "Last 1,000", "Copy full path", "requestFullscreen",
        "chart.js@4.4.7", "textContent",
    ):
        assert required in dashboard.HTML
