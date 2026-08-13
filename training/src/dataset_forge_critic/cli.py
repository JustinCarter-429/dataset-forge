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
        else:
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
    except (ValueError, ValidationError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
