"""Read-only live browser dashboard for one Dataset Forge Critic run."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _event_scalars(run_dir: Path) -> tuple[dict[str, list[dict[str, float]]], list[str]]:
    series: dict[str, dict[int, dict[str, float]]] = {}
    errors: list[str] = []
    event_files = sorted((run_dir / "tensorboard").rglob("events.out.tfevents.*"))
    if not event_files:
        return {}, ["TensorBoard events are not available yet."]
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}, ["TensorBoard is not installed in this environment."]
    for event_file in event_files:
        try:
            accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            accumulator.Reload()
            for tag in accumulator.Tags().get("scalars", []):
                points = series.setdefault(tag, {})
                for event in accumulator.Scalars(tag):
                    candidate = {"step": int(event.step), "value": float(event.value), "wall_time": float(event.wall_time)}
                    current = points.get(int(event.step))
                    if current is None or candidate["wall_time"] >= current["wall_time"]:
                        points[int(event.step)] = candidate
        except Exception as exc:  # A partially-written event must not take down monitoring.
            errors.append(f"Delayed/unreadable event file {event_file.name}: {type(exc).__name__}")
    return {tag: [points[step] for step in sorted(points)] for tag, points in series.items()}, errors


def _pick(series: dict[str, list[dict[str, float]]], *tags: str) -> list[dict[str, float]]:
    for tag in tags:
        if series.get(tag):
            return series[tag]
    return []


def _downsample(points: list[dict[str, float]], limit: int = 600) -> list[dict[str, float]]:
    if len(points) <= limit:
        return points
    stride = math.ceil(len(points) / limit)
    sampled = points[::stride]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def _rate(points: list[dict[str, float]], current_step: int, elapsed_seconds: float | None) -> tuple[float | None, str | None, list[dict[str, float]]]:
    timed = [point for point in points if point["step"] >= 0 and point["wall_time"] > 0]
    recent = timed[-101:]  # Up to 100 observed step intervals.
    rate_points: list[dict[str, float]] = []
    chart_window = timed[-201:]
    for left, right in zip(chart_window, chart_window[1:]):
        step_delta = right["step"] - left["step"]
        time_delta = right["wall_time"] - left["wall_time"]
        if step_delta > 0 and time_delta > 0:
            rate_points.append({"step": right["step"], "value": 60.0 * step_delta / time_delta, "wall_time": right["wall_time"]})
    if len(recent) >= 51:
        step_delta = recent[-1]["step"] - recent[0]["step"]
        time_delta = recent[-1]["wall_time"] - recent[0]["wall_time"]
        if step_delta > 0 and time_delta > 0:
            return 60.0 * step_delta / time_delta, f"rolling {step_delta}-step rate", rate_points
    if current_step > 0 and elapsed_seconds and elapsed_seconds > 0:
        return 60.0 * current_step / elapsed_seconds, "full-run average", rate_points
    return None, None, rate_points


def _iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="seconds")


def build_snapshot(run_dir: Path, total_steps: int, *, now: float | None = None) -> dict[str, Any]:
    """Read a run atomically enough for monitoring; never write to it."""
    run_dir = run_dir.resolve()
    now = time.time() if now is None else now
    status_path = run_dir / "status.json"
    status = _read_json(status_path)
    result = _read_json(run_dir / "result.json")
    lock = _read_json(run_dir / "run.lock")
    checkpoint = _read_json(run_dir / "latest-checkpoint.json")
    scalars, errors = _event_scalars(run_dir)
    loss_points = _pick(scalars, "train/loss", "loss")
    learning_rate_points = _pick(scalars, "train/learning_rate", "learning_rate")
    timing_points = loss_points or learning_rate_points

    event_step = max((int(point["step"]) for point in timing_points), default=0)
    current_step = max(int(status.get("step", 0) or 0), int(result.get("step", 0) or 0), event_step)
    remaining_steps = max(total_steps - current_step, 0)
    completion = min(max(100.0 * current_step / total_steps, 0.0), 100.0) if total_steps > 0 else 0.0

    elapsed = float(status.get("elapsed_seconds", 0) or 0) or None
    status_mtime = status_path.stat().st_mtime if status_path.exists() else None
    started_at = float(lock.get("started_at", 0) or 0) or None
    if elapsed and status_mtime:
        observed_start = status_mtime - elapsed
        started_at = min(started_at, observed_start) if started_at else observed_start
    elif timing_points:
        started_at = min(point["wall_time"] for point in timing_points)
    if started_at:
        elapsed = max(now - started_at, elapsed or 0.0)

    steps_per_minute, rate_basis, rate_points = _rate(timing_points, current_step, elapsed)
    eta_seconds = None
    if remaining_steps == 0:
        eta_seconds = 0.0
    elif steps_per_minute and steps_per_minute > 0:
        eta_seconds = 60.0 * remaining_steps / steps_per_minute

    pid = int(lock.get("pid", 0) or 0)
    result_status = str(result.get("status", "")).upper()
    if current_step >= total_steps or result_status == "TRAINING_BUDGET_COMPLETE":
        process_status = "completed"
    elif result_status == "GRACEFULLY_STOPPED" or ((run_dir / "STOP_REQUESTED").exists() and not _process_alive(pid)):
        process_status = "stopped"
    elif pid and _process_alive(pid):
        process_status = "running"
    else:
        process_status = "failed"

    current_loss = float(status["loss"]) if status.get("loss") is not None else (loss_points[-1]["value"] if loss_points else None)
    recent_losses = [point["value"] for point in loss_points[-20:]]
    smoothed_loss = sum(recent_losses) / len(recent_losses) if recent_losses else None
    learning_rate = float(status["learning_rate"]) if status.get("learning_rate") is not None else (learning_rate_points[-1]["value"] if learning_rate_points else None)

    return {
        "run_dir": str(run_dir), "current_step": current_step, "total_steps": total_steps,
        "remaining_steps": remaining_steps, "completion_percent": completion,
        "started_at": _iso(started_at), "elapsed_seconds": elapsed,
        "eta_seconds": eta_seconds, "eta_basis": rate_basis,
        "estimated_completion": _iso(now + eta_seconds) if eta_seconds is not None else None,
        "current_loss": current_loss, "smoothed_loss": smoothed_loss,
        "learning_rate": learning_rate, "steps_per_minute": steps_per_minute,
        "latest_checkpoint": checkpoint.get("path"), "process_status": process_status, "pid": pid or None,
        "charts": {"loss": _downsample(loss_points), "learning_rate": _downsample(learning_rate_points), "step_rate": _downsample(rate_points)},
        "warnings": errors, "refreshed_at": _iso(now),
    }


HTML = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dataset Forge Critic — Live Training</title><style>
:root{color-scheme:dark;--bg:#090d16;--panel:#111827;--line:#263247;--muted:#92a0b7;--text:#f4f7fb;--cyan:#35d0ff;--violet:#9b87f5;--green:#39d98a;--red:#ff6b7a}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#13213b 0,#090d16 38%);color:var(--text);font:15px system-ui,sans-serif}.wrap{max-width:1500px;margin:auto;padding:28px}.top{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}.eyebrow{color:var(--cyan);font-weight:750;letter-spacing:.12em;text-transform:uppercase;font-size:12px}h1{margin:7px 0 5px;font-size:30px}.sub{color:var(--muted);word-break:break-all}.badge{padding:9px 14px;border:1px solid var(--line);border-radius:999px;font-weight:750;text-transform:uppercase}.running{color:var(--green)}.completed{color:var(--cyan)}.failed,.stopped{color:var(--red)}.progress-shell{margin:25px 0;background:#070a10;border:1px solid var(--line);border-radius:15px;padding:15px}.progress-meta{display:flex;justify-content:space-between;margin-bottom:9px}.bar{height:18px;background:#202a3a;border-radius:99px;overflow:hidden}.fill{height:100%;width:0;background:linear-gradient(90deg,var(--violet),var(--cyan));transition:width .4s}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}.card,.chart{background:rgba(17,24,39,.92);border:1px solid var(--line);border-radius:15px;padding:16px}.label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.07em}.value{font-size:25px;font-weight:760;margin-top:7px;overflow-wrap:anywhere}.small{font-size:16px}.charts{display:grid;grid-template-columns:2fr 1fr;gap:14px;margin-top:14px}.chart.primary{grid-row:span 2}.chart h2{font-size:16px;margin:0 0 12px}svg{width:100%;height:260px;display:block}.secondary svg{height:180px}.empty{height:180px;display:grid;place-items:center;color:var(--muted)}.foot{color:var(--muted);margin-top:14px;font-size:13px}@media(max-width:900px){.charts{grid-template-columns:1fr}.chart.primary{grid-row:auto}.top{display:block}.badge{display:inline-block;margin-top:12px}}
</style></head><body><main class="wrap"><div class="top"><div><div class="eyebrow">Dataset Forge Critic</div><h1>Live Training Dashboard</h1><div class="sub" id="run"></div></div><div class="badge" id="status">Loading</div></div>
<section class="progress-shell"><div class="progress-meta"><b id="progressText">—</b><span id="percent">—</span></div><div class="bar"><div class="fill" id="fill"></div></div></section>
<section class="cards"><div class="card"><div class="label">Current step</div><div class="value" id="step">—</div></div><div class="card"><div class="label">Remaining</div><div class="value" id="remaining">—</div></div><div class="card"><div class="label">Current loss</div><div class="value" id="loss">—</div></div><div class="card"><div class="label">Smoothed loss · 20 pts</div><div class="value" id="smooth">—</div></div><div class="card"><div class="label">Learning rate</div><div class="value" id="lr">—</div></div><div class="card"><div class="label">Recent steps/min</div><div class="value" id="rate">—</div></div><div class="card"><div class="label">Started</div><div class="value small" id="started">—</div></div><div class="card"><div class="label">Elapsed</div><div class="value small" id="elapsed">—</div></div><div class="card"><div class="label">ETA · estimate</div><div class="value small" id="eta">Insufficient data</div></div><div class="card"><div class="label">Estimated completion</div><div class="value small" id="finish">Insufficient data</div></div><div class="card"><div class="label">Latest checkpoint</div><div class="value small" id="checkpoint">—</div></div></section>
<section class="charts"><div class="chart primary"><h2>Training loss</h2><div id="lossChart"></div></div><div class="chart secondary"><h2>Learning rate</h2><div id="lrChart"></div></div><div class="chart secondary"><h2>Recent step rate</h2><div id="rateChart"></div></div></section><div class="foot" id="foot">Refreshing every five seconds.</div></main>
<script>
const $=id=>document.getElementById(id), fmt=(v,d=4)=>v==null?'—':Number(v).toFixed(d), dur=s=>{if(s==null)return'—';s=Math.max(0,Math.round(s));let d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60),x=s%60;return[d?d+'d':null,h?h+'h':null,m?m+'m':null,x+'s'].filter(Boolean).join(' ')}, when=v=>v?new Date(v).toLocaleString():'—';
function chart(id,pts,color,format){const el=$(id);if(!pts||pts.length<2){el.innerHTML='<div class="empty">Waiting for metric data</div>';return}const w=900,h=260,p=35,x0=pts[0].step,x1=pts[pts.length-1].step||x0+1,ys=pts.map(p=>p.value),y0=Math.min(...ys),y1=Math.max(...ys);if(y1===y0)y1=y0+1;const xy=pts.map(q=>`${p+(q.step-x0)/(x1-x0)*(w-2*p)},${h-p-(q.value-y0)/(y1-y0)*(h-2*p)}`).join(' ');el.innerHTML=`<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><line x1="${p}" y1="${h-p}" x2="${w-p}" y2="${h-p}" stroke="#263247"/><line x1="${p}" y1="${p}" x2="${p}" y2="${h-p}" stroke="#263247"/><polyline points="${xy}" fill="none" stroke="${color}" stroke-width="3" vector-effect="non-scaling-stroke"/><text x="${p}" y="${h-8}" fill="#92a0b7">step ${x0}</text><text x="${w-p}" y="${h-8}" text-anchor="end" fill="#92a0b7">step ${x1}</text><text x="${p+5}" y="${p+14}" fill="#92a0b7">${format(y1)}</text><text x="${p+5}" y="${h-p-7}" fill="#92a0b7">${format(y0)}</text></svg>`}
async function refresh(){try{const d=await fetch('/api/status',{cache:'no-store'}).then(r=>r.json());$('run').textContent=d.run_dir;$('status').textContent=d.process_status+(d.pid?' · PID '+d.pid:'');$('status').className='badge '+d.process_status;$('step').textContent=d.current_step.toLocaleString()+' / '+d.total_steps.toLocaleString();$('remaining').textContent=d.remaining_steps.toLocaleString();$('percent').textContent=d.completion_percent.toFixed(2)+'%';$('progressText').textContent=d.current_step.toLocaleString()+' of '+d.total_steps.toLocaleString()+' optimizer steps';$('fill').style.width=d.completion_percent+'%';$('loss').textContent=fmt(d.current_loss,5);$('smooth').textContent=fmt(d.smoothed_loss,5);$('lr').textContent=d.learning_rate==null?'—':Number(d.learning_rate).toExponential(3);$('rate').textContent=d.steps_per_minute==null?'—':fmt(d.steps_per_minute,2);$('started').textContent=when(d.started_at);$('elapsed').textContent=dur(d.elapsed_seconds);$('eta').textContent=d.eta_seconds==null?'Insufficient data':dur(d.eta_seconds)+(d.eta_basis?' · '+d.eta_basis:'');$('finish').textContent=d.estimated_completion?when(d.estimated_completion):'Insufficient data';$('checkpoint').textContent=d.latest_checkpoint||'None yet';chart('lossChart',d.charts.loss,'#35d0ff',v=>Number(v).toFixed(4));chart('lrChart',d.charts.learning_rate,'#9b87f5',v=>Number(v).toExponential(2));chart('rateChart',d.charts.step_rate,'#39d98a',v=>Number(v).toFixed(2)+'/m');$('foot').textContent='Last refresh: '+when(d.refreshed_at)+(d.warnings.length?' · '+d.warnings.join(' · '):'');}catch(e){$('foot').textContent='Refresh delayed: '+e}}
refresh();setInterval(refresh,5000);
</script></body></html>"""


def create_server(run_dir: Path, host: str, port: int, total_steps: int) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("LIVE_DASHBOARD_LOOPBACK_ONLY")
    if total_steps <= 0:
        raise ValueError("TOTAL_STEPS_MUST_BE_POSITIVE")
    resolved = run_dir.resolve()
    if not resolved.is_dir():
        raise ValueError(f"RUN_DIR_NOT_FOUND:{resolved}")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/status":
                payload = json.dumps(build_snapshot(resolved, total_steps), separators=(",", ":")).encode()
                content_type = "application/json"
            elif path in {"/", "/index.html"}:
                payload = HTML.encode()
                content_type = "text/html; charset=utf-8"
            elif path == "/healthz":
                payload = b'{"status":"ok"}'
                content_type = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"dashboard {self.address_string()} {format % args}")

    return ThreadingHTTPServer(("127.0.0.1" if host == "localhost" else host, port), Handler)


def serve(run_dir: Path, host: str = "127.0.0.1", port: int = 7007, total_steps: int = 14561) -> None:
    server = create_server(run_dir, host, port, total_steps)
    print(f"Dataset Forge live dashboard: http://{host}:{server.server_port} (read-only, run={run_dir.resolve()})", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Dataset Forge Critic live dashboard")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7007)
    parser.add_argument("--total-steps", type=int, required=True)
    args = parser.parse_args(argv)
    serve(args.run_dir, args.host, args.port, args.total_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
