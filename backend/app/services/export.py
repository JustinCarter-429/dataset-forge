import csv
import json
from pathlib import Path
from typing import Protocol
from ..domain.enums import OutputFormat
from ..domain.models import CanonicalDataset


class DatasetExporter(Protocol):
    def export(self, dataset: CanonicalDataset, output_format: OutputFormat, destination: Path) -> Path: ...


class DeterministicDatasetExporter:
    @staticmethod
    def _safe_cell(value: str) -> str:
        # Keep CSV deterministic while preventing spreadsheet formula execution.
        return "'" + value if value.startswith(("=", "+", "-", "@")) else value

    def export(self, dataset: CanonicalDataset, output_format: OutputFormat, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if output_format == OutputFormat.JSON:
            destination.write_text(json.dumps([record.model_dump() for record in dataset.records], indent=2), encoding="utf-8")
        elif output_format == OutputFormat.JSONL:
            with destination.open("w", encoding="utf-8", newline="\n") as handle:
                for record in dataset.records:
                    handle.write(json.dumps(record.model_dump(), ensure_ascii=False, separators=(",", ":")) + "\n")
        else:
            with destination.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["instruction", "input", "output", "metadata", "context", "expected_output", "category", "difficulty", "source_refs", "evidence"])
                writer.writeheader()
                for record in dataset.records:
                    row = record.model_dump()
                    row["instruction"] = self._safe_cell(row["instruction"])
                    row["input"] = self._safe_cell(row["input"])
                    row["output"] = self._safe_cell(row["output"])
                    row["context"] = self._safe_cell(row["context"])
                    row["expected_output"] = self._safe_cell(row["expected_output"])
                    row["source_refs"] = json.dumps(row["source_refs"], sort_keys=True)
                    row["evidence"] = json.dumps([item.model_dump() if hasattr(item, "model_dump") else item for item in row["evidence"]], sort_keys=True)
                    row["metadata"] = json.dumps(row["metadata"], sort_keys=True)
                    writer.writerow(row)
        return destination
