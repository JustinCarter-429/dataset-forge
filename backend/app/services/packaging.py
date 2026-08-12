import json
from pathlib import Path
from typing import Protocol
from zipfile import ZIP_DEFLATED, ZipFile
from ..domain.models import GenerationManifest


class DatasetPackager(Protocol):
    def package(self, exported_file: Path, manifest: GenerationManifest, destination: Path, validation_report: dict | None = None, quality_review: dict | None = None, dataset_spec: dict | None = None) -> Path: ...


class ZipDatasetPackager:
    def package(self, exported_file: Path, manifest: GenerationManifest, destination: Path, validation_report: dict | None = None, quality_review: dict | None = None, dataset_spec: dict | None = None) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
            archive.write(exported_file, exported_file.name)
            manifest_payload = manifest.model_dump(mode="json")
            report = dict(validation_report or {})
            quality = report.get("quality") or report
            final_summary = {**manifest.model_dump(mode="json", by_alias=True), "groundingStatus": "passed" if manifest.grounding_evaluated and manifest.grounding_failure_count == 0 else "failed" if manifest.grounding_evaluated else "not_evaluated", "reviewRecommendation": (quality_review or {}).get("recommendation"), "reviewFindingVerification": (quality_review or {}).get("reviewFindingVerification"), "revisionAttempted": bool(manifest.revision_round_count), "revisionSucceeded": bool(manifest.revision_round_count), "revisionSkipReason": (quality_review or {}).get("revisionSkipReason")}
            metadata = {"datasetName": "Dataset Forge dataset", "sourceFile": manifest.source_file, "format": manifest.requested_format.value, "generatedAt": manifest.generated_at.isoformat(), "finalSummary": final_summary, **final_summary}
            report["finalSummary"] = final_summary
            if quality_review is not None: quality_review = {**quality_review, "finalSummary": final_summary}
            if dataset_spec is not None:
                archive.writestr("dataset-spec.json", json.dumps(dataset_spec, indent=2, ensure_ascii=False) + "\n")
            if validation_report is not None:
                archive.writestr("validation-report.json", json.dumps(validation_report, indent=2))
                archive.writestr("quality-report.json", json.dumps(validation_report, indent=2))
            if quality_review is not None:
                archive.writestr("quality-review.json", json.dumps(quality_review, indent=2))
            archive.writestr("generation_manifest.json", json.dumps(manifest_payload, indent=2))
            archive.writestr("manifest.json", json.dumps(manifest_payload, indent=2))
            archive.writestr("metadata.json", json.dumps(metadata, indent=2))
            archive.writestr("README.txt", "Dataset Forge package. Dataset generation was authored by Carter 1.0 using the configured model runtime. Deterministic Dataset Forge validation remains authoritative for schema, grounding, security, duplication, and export eligibility. AI quality review is advisory and bounded. See validation-report.json and quality-review.json for details.\n")
        return destination
