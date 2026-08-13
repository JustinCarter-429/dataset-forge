"""Offline-only command-line validation utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from .config import load_dataset_registry, load_model_configuration
from .validation import validate_record
from .ingestion import inspect_local_datasets
from .transformation import transform_all
from .curation import curate
from .native import generate
from .native_v2 import build
from .gemma import preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dataset Forge Critic local validation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    config_parser = subparsers.add_parser("validate-config", help="Validate local model and dataset YAML")
    config_parser.add_argument("--config-dir", type=Path, default=Path(__file__).resolve().parents[2] / "configs")
    record_parser = subparsers.add_parser("validate-record", help="Validate a canonical record JSON file")
    record_parser.add_argument("path", type=Path)
    inspect_parser = subparsers.add_parser("inspect-local-datasets", help="Safely ingest and inspect local source datasets")
    inspect_parser.add_argument("--source-root", type=Path, required=True)
    transform_parser = subparsers.add_parser("transform-datasets", help="Transform locally ingested datasets to canonical JSONL")
    transform_parser.add_argument("--all", action="store_true", help="Transform all supported local datasets")
    transform_parser.add_argument("--dataset", action="append", choices=["google_facts_grounding", "halu_eval", "nvidia_helpsteer2", "openbmb_ultrafeedback"])
    transform_parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    transform_parser.add_argument("--validate-only", action="store_true")
    for name in ("analyze-corpus", "curate-corpus", "validate-curated-corpus"):
        item=subparsers.add_parser(name); item.add_argument("--workspace",type=Path,default=Path(__file__).resolve().parents[3]); item.add_argument("--config",type=Path,default=Path(__file__).resolve().parents[2]/"configs"/"curation.yaml"); item.add_argument("--dry-run",action="store_true"); item.add_argument("--force-rebuild",action="store_true")
    native_parser=subparsers.add_parser('generate-native-rule-corpus'); native_parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[3]); native_parser.add_argument('--dry-run',action='store_true')
    v2=subparsers.add_parser('build-native-rule-corpus'); v2.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[3]); v2.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[2]/'configs'/'native-rule-v2.yaml'); v2.add_argument('--dry-run',action='store_true')
    subparsers.add_parser('training-preflight')
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-config":
            load_model_configuration(args.config_dir / "model.yaml")
            registry = load_dataset_registry(args.config_dir / "datasets.yaml")
            print(f"Configuration valid: model.yaml and {len(registry.datasets)} dataset registry entries.")
        elif args.command == "validate-record":
            payload = json.loads(args.path.read_text(encoding="utf-8"))
            validate_record(payload)
            print(f"Canonical record valid: {args.path}")
        elif args.command == "inspect-local-datasets":
            results = inspect_local_datasets(args.source_root, Path(__file__).resolve().parents[2])
            print(json.dumps({key: value.get("snapshot_id") or value["status"] for key, value in results.items()}, sort_keys=True))
        elif args.command == "transform-datasets":
            if not args.all and not args.dataset:
                parser.error("transform-datasets requires --all or --dataset")
            results = transform_all(args.workspace, args.dataset if not args.all else None, validate_only=args.validate_only)
            report_root = args.workspace / "training" / "reports" / "transforms"
            manifest_root = args.workspace / "training" / "manifests" / "transforms"
            if not args.validate_only:
                report_root.mkdir(parents=True, exist_ok=True)
                manifest_root.mkdir(parents=True, exist_ok=True)
                for result in results:
                    (report_root / f"{result.dataset}.json").write_text(json.dumps(result.report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                    (manifest_root / f"{result.dataset}.json").write_text(json.dumps(result.manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({result.dataset: result.report["canonical_records"] for result in results}, sort_keys=True))
        elif args.command == 'generate-native-rule-corpus': print(json.dumps(generate(args.workspace,args.dry_run),sort_keys=True))
        elif args.command == 'build-native-rule-corpus': print(json.dumps(build(args.workspace,args.config,args.dry_run),sort_keys=True))
        elif args.command == 'training-preflight': print(json.dumps(preflight(),sort_keys=True))
        else:
            result=curate(args.workspace,args.config,dry_run=args.dry_run or args.command=="analyze-corpus")
            print(json.dumps(result,sort_keys=True))
    except (ValueError, ValidationError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
