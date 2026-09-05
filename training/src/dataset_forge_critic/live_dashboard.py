"""Polished, read-only live browser dashboard for one Critic training run."""
from __future__ import annotations

import argparse, json, math, os, statistics, time
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
    try:
        if pid <= 0: return False
        os.kill(pid, 0); return True
    except OSError: return False


def _event_scalars(run_dir: Path) -> tuple[dict[str, list[dict[str, float]]], list[str]]:
    series: dict[str, dict[int, dict[str, float]]] = {}; errors: list[str] = []
    files = sorted((run_dir / "tensorboard").rglob("events.out.tfevents.*"))
    if not files: return {}, ["TensorBoard events are not available yet."]
    try: from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError: return {}, ["TensorBoard is not installed in this environment."]
    for file in files:
        try:
            reader = EventAccumulator(str(file), size_guidance={"scalars": 0}); reader.Reload()
            for tag in reader.Tags().get("scalars", []):
                target = series.setdefault(tag, {})
                for event in reader.Scalars(tag):
                    point = {"step": int(event.step), "value": float(event.value), "wall_time": float(event.wall_time)}
                    if int(event.step) not in target or point["wall_time"] >= target[int(event.step)]["wall_time"]: target[int(event.step)] = point
        except Exception as exc: errors.append(f"Delayed/unreadable event file {file.name}: {type(exc).__name__}")
    return {tag: [points[step] for step in sorted(points)] for tag, points in series.items()}, errors


def _pick(series: dict[str, list[dict[str, float]]], *tags: str) -> tuple[list[dict[str, float]], str | None]:
    for tag in tags:
        if series.get(tag): return series[tag], f"TensorBoard scalar {tag}"
    return [], None


def _moving_average(values: list[float], window: int) -> list[float]:
    result=[]; total=0.0
    for index, value in enumerate(values):
        total += value
        if index >= window: total -= values[index-window]
        result.append(total/min(index+1, window))
    return result


def _downsample(points: list[dict[str, Any]], limit: int = 1800) -> list[dict[str, Any]]:
    """Preserve each bucket's raw minimum and maximum, including spikes."""
    if len(points) <= limit: return points
    width = math.ceil(len(points)/max(limit//2, 1)); chosen=[points[0]]
    for start in range(1, len(points)-1, width):
        bucket=points[start:min(start+width, len(points)-1)]
        extrema={min(range(len(bucket)), key=lambda i:bucket[i]["value"]), max(range(len(bucket)), key=lambda i:bucket[i]["value"])}
        chosen.extend(bucket[i] for i in sorted(extrema))
    chosen.append(points[-1]); return chosen


def _loss_chart(points: list[dict[str, float]]) -> list[dict[str, Any]]:
    values=[p["value"] for p in points]; avgs={n:_moving_average(values,n) for n in (20,50,100)}
    return _downsample([{**p, **{f"smooth{n}":avgs[n][i] for n in avgs}} for i,p in enumerate(points)])


def _rate(points: list[dict[str,float]], step: int, elapsed: float|None):
    intervals=[]
    for left,right in zip(points,points[1:]):
        ds=right["step"]-left["step"]; dt=right["wall_time"]-left["wall_time"]
        if ds>0 and dt>0: intervals.append((int(right["step"]),dt/ds,float(right["wall_time"])))
    chart=[{"step":s,"value":60/sec,"wall_time":wall} for s,sec,wall in intervals[-500:]]
    if len(intervals)>=50:
        sample=[x[1] for x in intervals[-100:]]; sec=statistics.median(sample)
        return 60/sec,f"rolling median of {len(sample)} steps",chart,sec
    if step>0 and elapsed and elapsed>0:
        sec=elapsed/step; return 60/sec,"full-run average",chart,sec
    return None,None,chart,None


def _iso(value: float|None): return datetime.fromtimestamp(value).astimezone().isoformat(timespec="seconds") if value is not None else None


def _trend(values: list[float]) -> str:
    if len(values)<40: return "Insufficient data"
    smooth=_moving_average(values,20); old=statistics.mean(smooth[-40:-20]); new=statistics.mean(smooth[-20:]); change=(new-old)/max(abs(old),1e-12)
    return "Improving" if change < -.02 else "Rising" if change > .02 else "Stable"


def build_snapshot(run_dir: Path, total_steps: int, *, now: float|None=None) -> dict[str,Any]:
    """Read only the selected run; never writes or signals the trainer."""
    run_dir=run_dir.resolve(); now=time.time() if now is None else now
    status_path=run_dir/"status.json"; status=_read_json(status_path); result=_read_json(run_dir/"result.json")
    lock=_read_json(run_dir/"run.lock"); checkpoint=_read_json(run_dir/"latest-checkpoint.json"); manifest=_read_json(run_dir/"run-manifest.json")
    scalars,warnings=_event_scalars(run_dir); loss,loss_source=_pick(scalars,"train/loss","loss"); lr,lr_source=_pick(scalars,"train/learning_rate","learning_rate")
    supervised,supervised_source=_pick(scalars,"throughput/supervised_tokens_per_second"); timing=loss or lr
    event_step=max((int(p["step"]) for p in timing),default=0); step=max(int(status.get("step",0) or 0),int(result.get("step",0) or 0),event_step)
    remaining=max(total_steps-step,0); percent=min(max(100*step/total_steps,0),100) if total_steps>0 else 0
    last_event=max((p["wall_time"] for values in scalars.values() for p in values),default=None)
    elapsed=float(status.get("elapsed_seconds",0) or 0) or None; status_mtime=status_path.stat().st_mtime if status_path.exists() else None
    started=float(lock.get("started_at",0) or 0) or None
    if elapsed and status_mtime:
        inferred=status_mtime-elapsed; started=min(started,inferred) if started else inferred
    elif timing: started=min(p["wall_time"] for p in timing)
    if started: elapsed=max(now-started,elapsed or 0)
    per_min,basis,rate_points,seconds_per_step=_rate(timing,step,elapsed); eta=0.0 if remaining==0 else remaining*seconds_per_step if seconds_per_step else None
    pid=int(lock.get("pid",0) or 0); result_status=str(result.get("status","")).upper()
    if step>=total_steps or result_status=="TRAINING_BUDGET_COMPLETE": process="completed"
    elif result_status=="GRACEFULLY_STOPPED": process="stopped"
    elif pid and _process_alive(pid): process="running"
    elif result_status: process="failed"
    else: process="stopped"
    freshness=last_event or status_mtime; metric_age=max(now-freshness,0) if freshness else None; stale_after=max(120,min(900,5*seconds_per_step)) if seconds_per_step else 300
    data_status="Awaiting metrics" if freshness is None else "Awaiting new metrics" if metric_age>stale_after and process=="running" else "Metrics stale" if metric_age>stale_after else "Metrics updating"
    if loss: current_loss=loss[-1]["value"]
    elif status.get("loss") is not None: current_loss=float(status["loss"]); loss_source="status.json loss"
    else: current_loss=None
    loss_values=[p["value"] for p in loss]; smooth=statistics.mean(loss_values[-20:]) if loss_values else None; recent_low=min(loss_values[-100:]) if loss_values else None
    if lr: learning_rate=lr[-1]["value"]
    elif status.get("learning_rate") is not None: learning_rate=float(status["learning_rate"]); lr_source="status.json learning_rate"
    else: learning_rate=None
    checkpoint_path=checkpoint.get("path"); checkpoint_name=Path(str(checkpoint_path)).name if checkpoint_path else None; checkpoint_time=checkpoint.get("saved_at") or checkpoint.get("timestamp")
    if checkpoint_path and checkpoint_time is None:
        try: checkpoint_time=Path(str(checkpoint_path)).stat().st_mtime
        except OSError: checkpoint_time=None
    try: checkpoint_time=float(checkpoint_time) if checkpoint_time is not None else None
    except (TypeError,ValueError): checkpoint_time=None
    config=manifest.get("config") if isinstance(manifest.get("config"),dict) else {}; limits=config.get("limits") if isinstance(config.get("limits"),dict) else {}; model_config=config.get("model") if isinstance(config.get("model"),dict) else {}
    model=manifest.get("model_id") or model_config.get("model_id"); missing=[]
    for label,value in (("Training loss",current_loss),("Learning rate",learning_rate),("Recent step rate",per_min),("Latest checkpoint",checkpoint_path)):
        if value is None: missing.append(f"{label}: not present in the selected run yet.")
    sources={"Current loss":loss_source or "Unavailable","Smoothed loss":f"20-point mean of {loss_source}" if loss_source else "Unavailable","Learning rate":lr_source or "Unavailable","Step rate and ETA":basis or "Unavailable","Progress":"Maximum of status.json, result.json, and selected TensorBoard step","Process status":"run.lock PID probe and result.json","Checkpoint":"latest-checkpoint.json"}
    return {"run_dir":str(run_dir),"run_name":run_dir.name,"model":model,"config_digest":manifest.get("config_digest"),"max_examples":limits.get("max_examples"),"current_step":step,"total_steps":total_steps,"remaining_steps":remaining,"completion_percent":percent,"started_at":_iso(started),"elapsed_seconds":elapsed,"eta_seconds":eta,"eta_basis":basis,"estimated_completion":_iso(now+eta) if eta is not None else None,"current_loss":current_loss,"smoothed_loss":smooth,"recent_low_loss":recent_low,"loss_trend":_trend(loss_values),"learning_rate":learning_rate,"steps_per_minute":per_min,"latest_checkpoint":checkpoint_path,"latest_checkpoint_name":checkpoint_name,"checkpoint_timestamp":_iso(checkpoint_time),"checkpoint_age_seconds":max(now-checkpoint_time,0) if checkpoint_time else None,"process_status":process,"data_status":data_status,"pid":pid or None,"last_event_timestamp":_iso(last_event),"metric_age_seconds":metric_age,"metric_sources":sources,"missing_metrics":missing,"charts":{"loss":_loss_chart(loss),"learning_rate":_downsample(lr),"step_rate":_downsample(rate_points),"supervised_throughput":_downsample(supervised)},"supervised_throughput_source":supervised_source,"warnings":warnings,"refreshed_at":_iso(now)}


HTML=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dataset Forge Critic — Live Training Monitor</title><script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script><style>
:root{color-scheme:dark;--bg:#070b12;--panel:#0e1623;--panel2:#111c2c;--line:#223047;--text:#f2f6fc;--muted:#8fa0b7;--cyan:#36c9ef;--purple:#a78bfa;--green:#42d392;--amber:#f5b942;--red:#fb7185}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px Inter,system-ui,sans-serif;font-variant-numeric:tabular-nums}.shell{max-width:1760px;margin:auto;padding:24px 30px 40px}.top{display:flex;justify-content:space-between;gap:20px;margin-bottom:18px}.brand{display:flex;gap:12px;align-items:center}.mark{width:36px;height:36px;display:grid;place-items:center;border:1px solid #2b5870;border-radius:9px;color:var(--cyan);font-weight:800;background:#0b2130}.eyebrow{color:var(--cyan);font-size:11px;font-weight:750;letter-spacing:.12em;text-transform:uppercase}h1{margin:3px 0 0;font-size:25px}.meta,.controls{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;align-items:center}.controls{justify-content:flex-end;margin:0}.chip{display:flex;align-items:center;gap:7px;padding:6px 9px;border:1px solid var(--line);background:#0a111c;border-radius:7px;max-width:350px}.chip span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.dot{width:8px;height:8px;border-radius:50%;background:var(--muted)}.good{background:var(--green)}.warn{background:var(--amber)}.bad{background:var(--red)}button,select{border:1px solid #2b3b55;background:#111b2a;color:var(--text);border-radius:8px;padding:8px 11px;font:inherit;cursor:pointer}.notice,.error{padding:10px 13px;margin-bottom:14px;border:1px solid #183d50;border-radius:10px;background:#091721;color:#b6dded}.notice{display:flex;justify-content:space-between}.error{display:none;border-color:#6e4c1d;background:#241b0d;color:#ffd98b;white-space:pre-wrap}.panel{background:var(--panel);border:1px solid var(--line);border-radius:13px}.progress{padding:22px 24px;margin-bottom:14px}.progress-grid{display:grid;grid-template-columns:1.4fr 1fr;gap:34px;align-items:center}.label{color:var(--muted);font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase}.big{font-size:48px;font-weight:770;line-height:1;letter-spacing:-.04em}.big small{font-size:17px;color:var(--muted);letter-spacing:0}.percent{font-size:30px;font-weight:750;color:var(--cyan);margin-top:10px}.muted{color:var(--muted)}.track{height:13px;margin-top:23px;border-radius:99px;background:#1a2638;border:1px solid #26364d}.fill{height:100%;width:0;border-radius:99px;background:var(--cyan)}.milestones{position:relative;height:28px}.mile{position:absolute;transform:translateX(-50%);top:8px;color:#74859d;font-size:11px}.mile:before{content:"";position:absolute;left:50%;top:-22px;height:8px;border-left:1px solid #5d718d}.progress-stats{display:grid;grid-template-columns:1fr 1fr;gap:18px 24px}.value{margin-top:6px;font-size:20px;font-weight:690;overflow:hidden;text-overflow:ellipsis}.helper{grid-column:1/-1;color:#718198;font-size:12px;border-top:1px solid var(--line);padding-top:12px}.kpis{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin-bottom:14px}.kpi{padding:15px;min-height:104px}.kpi .value{font-size:22px;white-space:nowrap}.sub{color:#718198;font-size:11px;margin-top:7px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.chart-panel{padding:18px;margin-bottom:14px}.panelhead{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:14px}.panelhead h2{margin:0;font-size:17px}.titleline{display:flex;align-items:center;gap:10px}.trend{padding:5px 8px;border:1px solid #2b3b55;border-radius:6px;color:var(--muted);font-size:12px}.trend.improving{color:var(--green);border-color:#235f49}.trend.rising{color:var(--amber);border-color:#6c5222}.chartbox{height:400px;position:relative}.secondary{display:grid;grid-template-columns:1fr 1fr;gap:14px}.secondary .chartbox{height:250px}.chart-panel:fullscreen{background:var(--bg);padding:28px}.chart-panel:fullscreen .chartbox{height:calc(100vh - 110px)}details{margin-top:14px;padding:0 18px}summary{cursor:pointer;padding:16px 0;font-weight:700}.details{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:15px;padding-bottom:18px}.detail-value{margin-top:5px;color:#cbd6e6;overflow-wrap:anywhere}.sources{grid-column:1/-1;border-top:1px solid var(--line);padding-top:13px}.source{display:grid;grid-template-columns:180px 1fr;gap:10px;padding:4px 0;color:#a9b6c8}.footer{display:flex;justify-content:space-between;gap:12px;color:#697991;font-size:12px;margin-top:16px}@media(max-width:1200px){.kpis{grid-template-columns:repeat(3,1fr)}.progress-grid{grid-template-columns:1fr}.secondary{grid-template-columns:1fr}.details{grid-template-columns:1fr 1fr}}@media(max-width:760px){.shell{padding:16px}.top{display:block}.controls{justify-content:flex-start;margin-top:14px}.kpis{grid-template-columns:1fr 1fr}.progress-stats,.details{grid-template-columns:1fr}.big{font-size:37px}.chartbox{height:330px}.panelhead{align-items:flex-start;flex-direction:column}.notice{display:block}.source{grid-template-columns:1fr}}
</style></head><body><main class="shell"><header class="top"><div><div class="brand"><div class="mark">DF</div><div><div class="eyebrow">Dataset Forge Critic</div><h1>Live Training Monitor</h1></div></div><div class="meta"><div class="chip">Run <span id="runName">—</span></div><div class="chip">Model <span id="model">—</span></div><div class="chip"><i class="dot" id="pDot"></i><span id="pHead">Loading</span></div><div class="chip">PID <span id="pid">—</span></div></div></div><div class="controls"><div class="chip"><i class="dot" id="dDot"></i><span id="data">Connecting</span></div><label><input id="auto" type="checkbox" checked> Auto refresh</label><button id="pause">Pause charts</button><button id="refresh">Refresh</button></div></header><div class="notice"><span>Monitoring only — training controls are disabled</span><span id="refreshed">Last refresh —</span></div><div class="error" id="error"></div>
<section class="panel progress"><div class="progress-grid"><div><div class="label">Training progress</div><div class="big"><span id="step">—</span> <small>/ <span id="total">—</span> optimizer steps</small></div><div class="percent" id="percent">—</div><div class="muted"><span id="remaining">—</span> steps remaining</div><div class="track"><div class="fill" id="fill"></div></div><div class="milestones"><span class="mile" style="left:25%">25%</span><span class="mile" style="left:50%">50%</span><span class="mile" style="left:75%">75%</span><span class="mile" style="left:100%">100%</span></div></div><div class="progress-stats"><div><div class="label">Elapsed</div><div class="value" id="pElapsed">—</div></div><div><div class="label">Estimated remaining</div><div class="value" id="pEta">—</div></div><div><div class="label">Estimated finish</div><div class="value" id="finish">—</div></div><div><div class="label">Recent speed</div><div class="value" id="pRate">—</div></div><div class="helper">ETA is estimated from recent training speed and may shift during checkpoints, evaluation, or delayed event writes.</div></div></div></section>
<section class="kpis" id="kpis"></section>
<section class="panel chart-panel" id="lossPanel"><div class="panelhead"><div class="titleline"><h2>Training loss</h2><span class="trend" id="trend">Insufficient data</span></div><div><label class="muted">Smoothing <select id="smoothing"><option value="raw">Raw</option><option value="20" selected>20 steps</option><option value="50">50 steps</option><option value="100">100 steps</option></select></label> <label class="muted">Range <select id="range"><option value="all">All</option><option value="100">Last 100</option><option value="500">Last 500</option><option value="1000" selected>Last 1,000</option></select></label> <button id="reset">Reset view</button> <button id="expand">Expand</button></div></div><div class="chartbox"><canvas id="lossChart"></canvas></div></section>
<section class="secondary"><section class="panel chart-panel"><div class="panelhead"><div><h2>Learning rate</h2><div class="muted" id="lrNow">Current —</div></div></div><div class="chartbox"><canvas id="lrChart"></canvas></div></section><section class="panel chart-panel"><div class="panelhead"><div><h2>Step rate</h2><div class="muted" id="rateNow">Current —</div></div></div><div class="chartbox"><canvas id="rateChart"></canvas></div></section><section class="panel chart-panel" id="tpPanel" hidden><div class="panelhead"><h2>Supervised-token throughput</h2></div><div class="chartbox"><canvas id="tpChart"></canvas></div></section></section>
<details class="panel"><summary>Run details and metric provenance</summary><div class="details"><div><div class="label">Run directory</div><div class="detail-value" id="runDir"></div></div><div><div class="label">Model identity</div><div class="detail-value" id="modelDetail"></div></div><div><div class="label">Configuration digest</div><div class="detail-value" id="digest"></div></div><div><div class="label">Maximum steps / examples</div><div class="detail-value" id="limits"></div></div><div><div class="label">Latest checkpoint path</div><div class="detail-value" id="checkpointPath"></div></div><div><div class="label">Checkpoint / event timestamps</div><div class="detail-value" id="timestamps"></div></div><div class="sources"><div class="label">Metric sources</div><div id="sources"></div><div class="label" style="margin-top:12px">Unavailable metrics</div><div class="detail-value" id="missing">None</div></div></div></details><footer class="footer"><span>Read-only selected-run view. Chart.js 4.4.7 loads from pinned jsDelivr CDN.</span><span>Refresh interval: 5 seconds</span></footer></main>
<script>
const $=id=>document.getElementById(id),text=(id,v)=>$(id).textContent=v;let latest,charts={},paused=false,timer;const fmt=(v,n=5)=>v==null?'—':Number(v).toLocaleString(undefined,{maximumFractionDigits:n}),dur=v=>{if(v==null)return'—';let s=Math.max(0,Math.round(v)),d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60),x=s%60;return[d?d+'d':0,h?h+'h':0,m?m+'m':0,(!d||!h)?x+'s':0].filter(Boolean).join(' ')},when=v=>v?new Date(v).toLocaleString(undefined,{dateStyle:'medium',timeStyle:'medium'})+' '+Intl.DateTimeFormat().resolvedOptions().timeZone:'—';
const axis=title=>({title:{display:true,text:title,color:'#8fa0b7'},ticks:{color:'#8191a7',maxTicksLimit:8},grid:{color:'rgba(111,132,160,.13)'},border:{color:'#32425a'}});function makeChart(id,sets,y){if(!window.Chart){$('error').style.display='block';text('error','Chart.js could not load. Metric cards continue updating.');return}return new Chart($(id),{type:'line',data:{datasets:sets},options:{responsive:true,maintainAspectRatio:false,animation:false,interaction:{mode:'index',intersect:false},plugins:{legend:{labels:{color:'#aebbd0',usePointStyle:true}},tooltip:{backgroundColor:'#07101b',borderColor:'#31445f',borderWidth:1,padding:11}},scales:{x:{...axis('Optimizer Step'),type:'linear'},y:axis(y)},elements:{point:{radius:0,hitRadius:8},line:{tension:.08}}}})}
function filtered(points){let r=$('range').value;if(r==='all'||!points.length)return points;let max=points.at(-1).step;return points.filter(p=>p.step>=max-Number(r))}function setChart(key,id,sets,y){if(!charts[key])charts[key]=makeChart(id,sets,y);else{charts[key].data.datasets=sets;charts[key].update('none')}}function renderCharts(d){if(paused)return;let n=$('smoothing').value,p=filtered(d.charts.loss||[]),raw={label:'Raw loss',data:p.map(x=>({x:x.step,y:x.value,wall:x.wall_time,raw:x.value,smooth:n==='raw'?null:x['smooth'+n]})),borderColor:n==='raw'?'#36c9ef':'rgba(54,201,239,.35)',borderWidth:n==='raw'?2:1};let sets=[raw];if(n!=='raw')sets.push({label:n+'-step smoothed',data:p.map(x=>({x:x.step,y:x['smooth'+n],wall:x.wall_time,raw:x.value,smooth:x['smooth'+n]})),borderColor:'#38a9ff',borderWidth:3});setChart('loss','lossChart',sets,'Training Loss');if(charts.loss)charts.loss.options.plugins.tooltip.callbacks={title:i=>'Optimizer step '+i[0].parsed.x,afterTitle:i=>i[0].raw.wall?'Recorded '+when(i[0].raw.wall*1000):'',label:i=>i.datasetIndex?' Smoothed loss: '+fmt(i.raw.smooth,7):' Raw loss: '+fmt(i.raw.raw,7)};let simple=(key,id,points,label,color,y)=>setChart(key,id,[{label,data:filtered(points||[]).map(x=>({x:x.step,y:x.value,wall:x.wall_time})),borderColor:color,borderWidth:2}],y);simple('lr','lrChart',d.charts.learning_rate,'Learning rate','#a78bfa','Learning Rate');simple('rate','rateChart',d.charts.step_rate,'Steps per minute','#42d392','Steps / Minute');let tp=d.charts.supervised_throughput||[];$('tpPanel').hidden=!tp.length;if(tp.length)simple('tp','tpChart',tp,'Supervised tokens / second','#f5b942','Tokens / Second')}
function render(d){latest=d;text('runName',d.run_name||'—');text('model',d.model||'Unknown');text('pid',d.pid||'—');text('pHead',d.process_status);$('pDot').className='dot '+(d.process_status==='running'||d.process_status==='completed'?'good':d.process_status==='failed'?'bad':'warn');text('data',d.data_status);$('dDot').className='dot '+(d.data_status==='Metrics updating'?'good':'warn');text('refreshed','Last refresh '+when(d.refreshed_at));text('step',Number(d.current_step).toLocaleString());text('total',Number(d.total_steps).toLocaleString());text('remaining',Number(d.remaining_steps).toLocaleString());text('percent',fmt(d.completion_percent,2)+'% complete');$('fill').style.width=d.completion_percent+'%';text('pElapsed',dur(d.elapsed_seconds));text('pEta',d.eta_seconds==null?'Insufficient data':dur(d.eta_seconds));text('finish',d.estimated_completion?when(d.estimated_completion):'Insufficient data');text('pRate',d.steps_per_minute==null?'Insufficient data':fmt(d.steps_per_minute,2)+' steps/min');const items=[['Current loss',fmt(d.current_loss,7),d.metric_sources['Current loss']],['Smoothed loss',fmt(d.smoothed_loss,7),'20-point moving mean'],['Recent low loss',fmt(d.recent_low_loss,7),'Lowest of last 100 points'],['Learning rate',d.learning_rate==null?'—':Number(d.learning_rate).toExponential(3),d.metric_sources['Learning rate']],['Steps / minute',fmt(d.steps_per_minute,2),d.eta_basis||'Insufficient timing data'],['Elapsed time',dur(d.elapsed_seconds),'Started '+when(d.started_at)],['Remaining time',d.eta_seconds==null?'Insufficient':dur(d.eta_seconds),'Estimate'],['Latest checkpoint',d.latest_checkpoint_name||'None yet','Copy full path'],['Checkpoint age',dur(d.checkpoint_age_seconds),when(d.checkpoint_timestamp)],['Process status',d.process_status,d.data_status]];let root=$('kpis');root.replaceChildren();items.forEach((x,i)=>{let card=document.createElement('div');card.className='panel kpi';let a=document.createElement('div'),b=document.createElement('div'),c=document.createElement('div');a.className='label';b.className='value';c.className='sub';a.textContent=x[0];b.textContent=x[1];c.textContent=x[2];if(i===7){b.title=d.latest_checkpoint||'';c.textContent='';let btn=document.createElement('button');btn.textContent='Copy full path';btn.onclick=async()=>{if(d.latest_checkpoint){await navigator.clipboard.writeText(d.latest_checkpoint);btn.textContent='Copied'}};c.append(btn)}card.append(a,b,c);root.append(card)});text('lrNow','Current '+(d.learning_rate==null?'—':Number(d.learning_rate).toExponential(3)));text('rateNow','Current '+fmt(d.steps_per_minute,2)+' steps/min');text('trend',d.loss_trend);$('trend').className='trend '+d.loss_trend.toLowerCase();text('runDir',d.run_dir);text('modelDetail',d.model||'Unavailable');text('digest',d.config_digest||'Unavailable');text('limits',d.total_steps.toLocaleString()+' steps / '+(d.max_examples==null?'unavailable':Number(d.max_examples).toLocaleString()+' examples'));text('checkpointPath',d.latest_checkpoint||'None yet');text('timestamps','Checkpoint: '+when(d.checkpoint_timestamp)+' · Last event: '+when(d.last_event_timestamp));let src=$('sources');src.replaceChildren();Object.entries(d.metric_sources).forEach(([a,b])=>{let row=document.createElement('div');row.className='source';let strong=document.createElement('strong'),span=document.createElement('span');strong.textContent=a;span.textContent=b;row.append(strong,span);src.append(row)});text('missing',d.missing_metrics.length?d.missing_metrics.join(' '):'None');$('error').style.display=d.warnings.length?'block':'none';text('error',d.warnings.join('\n'));renderCharts(d)}async function refresh(){try{let r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);render(await r.json())}catch(e){$('error').style.display='block';text('error','Dashboard refresh delayed: '+e.message)}}function schedule(){clearInterval(timer);if($('auto').checked)timer=setInterval(refresh,5000)}$('refresh').onclick=refresh;$('auto').onchange=schedule;$('pause').onclick=()=>{paused=!paused;text('pause',paused?'Resume charts':'Pause charts');if(!paused&&latest)renderCharts(latest)};$('smoothing').onchange=()=>latest&&renderCharts(latest);$('range').onchange=()=>latest&&renderCharts(latest);$('reset').onclick=()=>{$('range').value='all';latest&&renderCharts(latest)};$('expand').onclick=()=>document.fullscreenElement?document.exitFullscreen():$('lossPanel').requestFullscreen();refresh();schedule();
</script></body></html>'''


HTML = HTML.replace(
    "</style>",
    ".top{align-items:flex-start}.top>div:first-child{min-width:0;flex:1}"
    ".controls{max-width:560px;flex:0 1 auto}</style>",
    1,
)


def create_server(run_dir: Path, host: str, port: int, total_steps: int) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1","localhost"}: raise ValueError("LIVE_DASHBOARD_LOOPBACK_ONLY")
    if total_steps<=0: raise ValueError("TOTAL_STEPS_MUST_BE_POSITIVE")
    resolved=run_dir.resolve()
    if not resolved.is_dir(): raise ValueError(f"RUN_DIR_NOT_FOUND:{resolved}")
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path=urlparse(self.path).path
            if path=="/api/status": payload=json.dumps(build_snapshot(resolved,total_steps),separators=(",",":")).encode(); content_type="application/json"
            elif path in {"/","/index.html"}: payload=HTML.encode(); content_type="text/html; charset=utf-8"
            elif path=="/healthz": payload=b'{"status":"ok"}'; content_type="application/json"
            else: self.send_error(404); return
            self.send_response(200); self.send_header("Content-Type",content_type); self.send_header("Content-Length",str(len(payload))); self.send_header("Cache-Control","no-store"); self.send_header("X-Content-Type-Options","nosniff"); self.end_headers(); self.wfile.write(payload)
        def log_message(self,format,*args): print(f"dashboard {self.address_string()} {format % args}")
    return ThreadingHTTPServer(("127.0.0.1" if host=="localhost" else host,port),Handler)


def serve(run_dir:Path,host="127.0.0.1",port=7007,total_steps=14561):
    server=create_server(run_dir,host,port,total_steps); print(f"Dataset Forge live dashboard: http://{host}:{server.server_port} (read-only, run={run_dir.resolve()})",flush=True)
    try: server.serve_forever(poll_interval=.5)
    except KeyboardInterrupt: pass
    finally: server.server_close()


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description="Read-only Dataset Forge Critic live dashboard"); parser.add_argument("--run-dir",type=Path,required=True); parser.add_argument("--host",default="127.0.0.1"); parser.add_argument("--port",type=int,default=7007); parser.add_argument("--total-steps",type=int,required=True); args=parser.parse_args(argv); serve(args.run_dir,args.host,args.port,args.total_steps); return 0


if __name__=="__main__": raise SystemExit(main())
