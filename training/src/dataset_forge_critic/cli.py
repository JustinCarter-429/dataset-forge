"""Dataset Forge Critic data, training, evaluation, and run-control CLI."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from pydantic import ValidationError

from .artifacts import export_adapter, sync_artifacts
from .auditing import audit_tokens, verify_training_data
from .config import load_dataset_registry, load_model_configuration
from .curation import curate
from .evaluator import evaluate
from .gemma import preflight as legacy_preflight
from .gpu_bundle import build as build_gpu_bundle, verify as verify_gpu_bundle
from .handoff import build_handoff, verify_handoff
from .ingestion import inspect_local_datasets
from .integrity import derive_jsonl
from .modeling import load_processor
from .native import generate
from .native_v2 import build
from .runtime import request_stop, status
from .synthetic import run_synthetic
from .trainer import run_directory, train
from .training_config import TrainingConfiguration, load_training_configuration
from .transformation import transform_all
from .validation import validate_record
from .vagon_handoff import build as build_vagon_handoff

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent


def _configuration(path: Path) -> TrainingConfiguration:
    config = load_training_configuration(path)
    base = path.resolve().parent
    updates = {}
    for name, value in config.data.model_dump().items():
        item = Path(value); updates[name] = item if item.is_absolute() else (base / item).resolve()
    output = config.output_root if config.output_root.is_absolute() else (base / config.output_root).resolve()
    return config.model_copy(update={"data": config.data.model_copy(update=updates), "output_root": output})


def _hardware_preflight(config: TrainingConfiguration | None) -> dict:
    result = legacy_preflight(); result.update({"platform": platform.platform(), "machine": platform.machine(), "gpu_certification": "PENDING"})
    command = shutil.which("nvidia-smi")
    if command:
        query = subprocess.run([command, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        result["nvidia_smi"] = query.stdout.strip() if query.returncode == 0 else query.stderr.strip()
    if config:
        free = shutil.disk_usage(config.output_root.parent).free / 1024 ** 3
        result.update({"config_digest": config.digest(), "profile": config.profile, "free_disk_gb": free, "minimum_free_disk_gb": config.limits.min_free_disk_gb})
        if free < config.limits.min_free_disk_gb: result["readiness"] = "INSUFFICIENT_DISK"
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset Forge Critic operational CLI")
    commands = parser.add_subparsers(dest="command", required=True)
    item = commands.add_parser("validate-config"); item.add_argument("--config-dir", type=Path, default=ROOT / "configs"); item.add_argument("--training-config", type=Path)
    item = commands.add_parser("validate-record"); item.add_argument("path", type=Path)
    item = commands.add_parser("inspect-local-datasets"); item.add_argument("--source-root", type=Path, required=True)
    item = commands.add_parser("transform-datasets"); item.add_argument("--all", action="store_true"); item.add_argument("--dataset", action="append"); item.add_argument("--workspace", type=Path, default=WORKSPACE); item.add_argument("--validate-only", action="store_true")
    for name in ("analyze-corpus", "curate-corpus", "validate-curated-corpus"):
        item = commands.add_parser(name); item.add_argument("--workspace", type=Path, default=WORKSPACE); item.add_argument("--config", type=Path, default=ROOT / "configs" / "curation.yaml"); item.add_argument("--dry-run", action="store_true"); item.add_argument("--force-rebuild", action="store_true")
    item = commands.add_parser("generate-native-rule-corpus"); item.add_argument("--workspace", type=Path, default=WORKSPACE); item.add_argument("--dry-run", action="store_true")
    item = commands.add_parser("build-native-rule-corpus"); item.add_argument("--workspace", type=Path, default=WORKSPACE); item.add_argument("--config", type=Path, default=ROOT / "configs" / "native-rule-v2.yaml"); item.add_argument("--dry-run", action="store_true")
    item = commands.add_parser("preflight"); item.add_argument("--config", type=Path)
    commands.add_parser("training-preflight")
    item = commands.add_parser("verify-data"); item.add_argument("--config", type=Path, required=True)
    for name in ("audit-tokens", "token-audit"):
        item = commands.add_parser(name); item.add_argument("--config", type=Path, required=True); item.add_argument("--limit-per-file", type=int)
    item = commands.add_parser("derive-data"); item.add_argument("--source", type=Path, required=True); item.add_argument("--destination", type=Path, required=True); item.add_argument("--source-identity", required=True)
    item = commands.add_parser("train"); item.add_argument("--config", type=Path, required=True); item.add_argument("--confirm", required=True); item.add_argument("--resume", type=Path)
    item = commands.add_parser("resume"); item.add_argument("--config", type=Path, required=True); item.add_argument("--confirm", required=True); item.add_argument("--checkpoint", type=Path, required=True)
    item = commands.add_parser("evaluate"); item.add_argument("--config", type=Path, required=True); item.add_argument("--adapter", type=Path); item.add_argument("--broad", action="store_true")
    item = commands.add_parser("launch"); item.add_argument("--config", type=Path, required=True); item.add_argument("--confirm", required=True); item.add_argument("--resume", type=Path)
    item = commands.add_parser("status"); item.add_argument("--run-dir", type=Path, required=True)
    item = commands.add_parser("stop"); item.add_argument("--run-dir", type=Path, required=True)
    item = commands.add_parser("cpu-synthetic"); item.add_argument("--public", type=Path, required=True); item.add_argument("--native", type=Path, required=True); item.add_argument("--checkpoint", type=Path, required=True); item.add_argument("--steps", type=int, required=True); item.add_argument("--resume", action="store_true")
    item = commands.add_parser("export-adapter"); item.add_argument("--config", type=Path, required=True); item.add_argument("--checkpoint", type=Path, required=True); item.add_argument("--destination", type=Path, required=True)
    item = commands.add_parser("export"); item.add_argument("--config", type=Path, required=True); item.add_argument("--checkpoint", type=Path, required=True); item.add_argument("--destination", type=Path, required=True)
    item = commands.add_parser("launch-monitor"); item.add_argument("--root", type=Path, default=ROOT); item.add_argument("--jupyter-port", type=int, default=8888); item.add_argument("--tensorboard-port", type=int, default=6006)
    item = commands.add_parser("sync-artifacts"); item.add_argument("--source", type=Path, required=True); item.add_argument("--destination", required=True); item.add_argument("--visibility", choices=["private", "public"]); item.add_argument("--retries", type=int, default=3)
    item = commands.add_parser("build-gpu-bundle"); item.add_argument("--workspace", type=Path, default=WORKSPACE)
    item = commands.add_parser("verify-gpu-bundle"); item.add_argument("--bundle-root", type=Path, required=True)
    item = commands.add_parser("build-vagon-handoff"); item.add_argument("--workspace", type=Path, default=WORKSPACE); item.add_argument("--target", type=Path, required=True)
    item = commands.add_parser("build-handoff"); item.add_argument("--workspace", type=Path, default=WORKSPACE)
    item = commands.add_parser("verify-handoff"); item.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser(); args = parser.parse_args(argv)
    try:
        if args.command == "validate-config":
            load_model_configuration(args.config_dir / "model.yaml"); registry = load_dataset_registry(args.config_dir / "datasets.yaml")
            output = {"status": "CONFIG_VALID", "datasets": len(registry.datasets)}
            if args.training_config: output.update({"training_digest": _configuration(args.training_config).digest()})
        elif args.command == "validate-record": output = validate_record(json.loads(args.path.read_text(encoding="utf-8"))).model_dump(mode="json")
        elif args.command == "inspect-local-datasets": output = inspect_local_datasets(args.source_root, ROOT)
        elif args.command == "transform-datasets":
            if not args.all and not args.dataset: parser.error("transform-datasets requires --all or --dataset")
            values = transform_all(args.workspace, None if args.all else args.dataset, validate_only=args.validate_only)
            if not args.validate_only:
                report_root = args.workspace / "training" / "reports" / "transforms"
                manifest_root = args.workspace / "training" / "manifests" / "transforms"
                report_root.mkdir(parents=True, exist_ok=True); manifest_root.mkdir(parents=True, exist_ok=True)
                for value in values:
                    (report_root / f"{value.dataset}.json").write_text(json.dumps(value.report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                    (manifest_root / f"{value.dataset}.json").write_text(json.dumps(value.manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            output = {value.dataset: value.report["canonical_records"] for value in values}
        elif args.command in {"analyze-corpus", "curate-corpus", "validate-curated-corpus"}: output = curate(args.workspace, args.config, dry_run=args.dry_run or args.command == "analyze-corpus")
        elif args.command == "generate-native-rule-corpus": output = generate(args.workspace, args.dry_run)
        elif args.command == "build-native-rule-corpus": output = build(args.workspace, args.config, args.dry_run)
        elif args.command in {"preflight", "training-preflight"}: output = _hardware_preflight(_configuration(args.config) if getattr(args, "config", None) else None)
        elif args.command == "verify-data": output = verify_training_data(_configuration(args.config))
        elif args.command in {"audit-tokens", "token-audit"}:
            config = _configuration(args.config); output = audit_tokens(config, load_processor(config), limit_per_file=args.limit_per_file)
        elif args.command == "derive-data": output = derive_jsonl(args.source, args.destination, source_identity=args.source_identity)
        elif args.command == "train": output = train(_configuration(args.config), confirmation=args.confirm, resume=args.resume)
        elif args.command == "resume": output = train(_configuration(args.config), confirmation=args.confirm, resume=args.checkpoint)
        elif args.command == "evaluate": output = evaluate(_configuration(args.config), adapter=args.adapter, broad=args.broad)
        elif args.command == "launch":
            config = _configuration(args.config)
            if args.confirm != config.digest(): raise ValueError(f"CONFIRMATION_REQUIRED:{config.digest()}")
            current = status(run_directory(config))
            if current["active"]: raise ValueError(f"RUN_ALREADY_ACTIVE:{current['pid']}")
            run_directory(config).mkdir(parents=True, exist_ok=True); log = (run_directory(config) / "trainer.log").open("a", encoding="utf-8")
            command = [sys.executable, "-m", "dataset_forge_critic.cli", "train", "--config", str(args.config.resolve()), "--confirm", args.confirm]
            if args.resume: command += ["--resume", str(args.resume.resolve())]
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True); output = {"status": "LAUNCHED", "pid": process.pid, "run_dir": str(run_directory(config)), "log": log.name}
        elif args.command == "status": output = status(args.run_dir)
        elif args.command == "stop": output = request_stop(args.run_dir)
        elif args.command == "cpu-synthetic": output = run_synthetic(args.public, args.native, args.checkpoint, steps=args.steps, resume=args.resume)
        elif args.command in {"export-adapter", "export"}: output = export_adapter(args.checkpoint, args.destination, _configuration(args.config))
        elif args.command == "launch-monitor":
            monitor_dir = args.root.resolve() / "outputs" / "monitor"; monitor_dir.mkdir(parents=True, exist_ok=True)
            tensorboard_log = (monitor_dir / "tensorboard.log").open("a", encoding="utf-8")
            jupyter_log = (monitor_dir / "jupyter.log").open("a", encoding="utf-8")
            tensorboard = subprocess.Popen([sys.executable, "-m", "tensorboard.main", "--logdir", str(args.root.resolve() / "outputs"), "--host", "127.0.0.1", "--port", str(args.tensorboard_port)], stdout=tensorboard_log, stderr=subprocess.STDOUT, start_new_session=True)
            jupyter = subprocess.Popen([sys.executable, "-m", "jupyterlab", "--no-browser", "--ip", "127.0.0.1", "--port", str(args.jupyter_port), "--notebook-dir", str(args.root.resolve())], stdout=jupyter_log, stderr=subprocess.STDOUT, start_new_session=True)
            output = {"status": "MONITORS_LAUNCHED", "jupyter_pid": jupyter.pid, "tensorboard_pid": tensorboard.pid,
                      "jupyter_url": f"http://127.0.0.1:{args.jupyter_port}", "tensorboard_url": f"http://127.0.0.1:{args.tensorboard_port}",
                      "ssh_forward": f"ssh -L {args.jupyter_port}:127.0.0.1:{args.jupyter_port} -L {args.tensorboard_port}:127.0.0.1:{args.tensorboard_port} <gpu-host>"}
        elif args.command == "sync-artifacts": output = sync_artifacts(args.source, args.destination, visibility=args.visibility, retries=args.retries)
        elif args.command == "build-gpu-bundle": output = build_gpu_bundle(args.workspace)
        elif args.command == "verify-gpu-bundle": output = verify_gpu_bundle(args.bundle_root)
        elif args.command == "build-vagon-handoff": output = build_vagon_handoff(args.workspace, args.target)
        elif args.command == "build-handoff": output = build_handoff(args.workspace)
        elif args.command == "verify-handoff": output = verify_handoff(args.root)
        else: raise AssertionError(args.command)
        print(json.dumps(output, sort_keys=True, indent=2, default=str)); return 0
    except (ValueError, ValidationError, json.JSONDecodeError, OSError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__": raise SystemExit(main())
