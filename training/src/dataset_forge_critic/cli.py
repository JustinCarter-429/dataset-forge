"""Offline-only command-line validation utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from .config import load_dataset_registry, load_model_configuration
from .validation import validate_record
from .ingestion import inspect_local_datasets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dataset Forge Critic local validation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    config_parser = subparsers.add_parser("validate-config", help="Validate local model and dataset YAML")
    config_parser.add_argument("--config-dir", type=Path, default=Path(__file__).resolve().parents[2] / "configs")
    record_parser = subparsers.add_parser("validate-record", help="Validate a canonical record JSON file")
    record_parser.add_argument("path", type=Path)
    inspect_parser = subparsers.add_parser("inspect-local-datasets", help="Safely ingest and inspect local source datasets")
    inspect_parser.add_argument("--source-root", type=Path, required=True)
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
        else:
            results = inspect_local_datasets(args.source_root, Path(__file__).resolve().parents[2])
            print(json.dumps({key: value.get("snapshot_id") or value["status"] for key, value in results.items()}, sort_keys=True))
    except (ValueError, ValidationError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
