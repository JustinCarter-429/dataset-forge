import json
from pathlib import Path
from typing import Protocol
from zipfile import ZIP_DEFLATED, ZipFile
from ..domain.models import GenerationManifest


class DatasetPackager(Protocol):
    def package(self, exported_file: Path, manifest: GenerationManifest, destination: Path, validation_report: dict | None = None, quality_review: dict | None = None) -> Path: ...


class ZipDatasetPackager:
    def package(self, exported_file: Path, manifest: GenerationManifest, destination: Path, validation_report: dict | None = None, quality_review: dict | None = None) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
            archive.write(exported_file, exported_file.name)
            manifest_payload = manifest.model_dump(mode="json")
            report = validation_report or {}
            metadata = {"datasetName": "Dataset Forge dataset", "sourceFile": manifest.source_file, "recordCount": manifest.record_count, "format": manifest.requested_format.value, "generatedAt": manifest.generated_at.isoformat(), "schemaVersion": manifest.schema_version, "generationMode": manifest.phase, "provider": manifest.provider, "model": manifest.model, "promptVersion": manifest.prompt_version, "validationStatus": report.get("status", manifest.validation_status), "groundingStatus": (report.get("grounding") or {}).get("status", "not_evaluated"), "groundedRecords": (report.get("grounding") or {}).get("grounded_records", 0), "finalRecordCount": (report.get("quality") or {}).get("final_record_count", manifest.record_count), "exactDuplicatesRemoved": (report.get("quality") or {}).get("exact_duplicates_removed", 0), "nearDuplicateCount": (report.get("quality") or {}).get("near_duplicate_pairs", 0), "qualityReviewVersion": (quality_review or {}).get("reviewVersion"), "reviewProvider": (quality_review or {}).get("provider"), "reviewModel": (quality_review or {}).get("model"), "reviewPromptVersion": (quality_review or {}).get("promptVersion"), "reviewStatus": (quality_review or {}).get("status", "not_evaluated"), "revisionAttempted": (quality_review or {}).get("revisionAttempted", False), "revisionSucceeded": (quality_review or {}).get("revisionSucceeded", False), "revisionCount": 1 if (quality_review or {}).get("revisionAttempted") else 0}
            if validation_report is not None:
                archive.writestr("validation-report.json", json.dumps(validation_report, indent=2))
            if quality_review is not None:
                archive.writestr("quality-review.json", json.dumps(quality_review, indent=2))
            archive.writestr("generation_manifest.json", json.dumps(manifest_payload, indent=2))
            archive.writestr("manifest.json", json.dumps(manifest_payload, indent=2))
            archive.writestr("metadata.json", json.dumps(metadata, indent=2))
            archive.writestr("README.txt", "Dataset Forge Phase 5 package. Generation uses the configured RunPod gpt-oss-20b provider; schema, source references, evidence, and grounding remain authoritative application checks. AI quality review is advisory and bounded, and at most one controlled revision is permitted. See validation-report.json and quality-review.json for details.\n")
        return destination
