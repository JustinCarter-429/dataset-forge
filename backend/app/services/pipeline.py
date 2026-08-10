from pathlib import Path
from ..domain.models import GenerationManifest, GenerationResult, PipelineInput
from ..domain.extraction_models import CanonicalExtractedDocument
from .extraction import DocumentExtractor, ExtractionService
from .generation import DatasetGenerator, RunPodDatasetGenerator
from .validation import DatasetValidator, Phase4ValidationService, StructuralDatasetValidator
from .export import DatasetExporter, DeterministicDatasetExporter
from .packaging import DatasetPackager, ZipDatasetPackager
from .quality_review import QualityReviewService, QualityRevisionService, RevisionPolicy


class PipelineService:
    def __init__(self, output_directory: Path, extractor: DocumentExtractor | None = None, generator: DatasetGenerator | None = None, validator: DatasetValidator | None = None, exporter: DatasetExporter | None = None, packager: DatasetPackager | None = None, *, quality_review_enabled: bool = False, quality_reviewer=None, quality_reviser=None, revision_policy=None):
        self.output_directory = output_directory
        self.extractor = extractor or ExtractionService()
        self.generator = generator
        self.validator = validator or StructuralDatasetValidator()
        self.exporter = exporter or DeterministicDatasetExporter()
        self.packager = packager or ZipDatasetPackager()
        self.phase4_validator = Phase4ValidationService()
        self.quality_review_enabled = quality_review_enabled
        self.quality_reviewer = quality_reviewer
        self.quality_reviser = quality_reviser
        self.revision_policy = revision_policy or RevisionPolicy()

    def run(self, request: PipelineInput, extracted_document: CanonicalExtractedDocument | None = None, *, file_id: str | None = None, model: str = "gpt-oss-20b", on_progress=None, on_state=None) -> GenerationResult:
        job_directory = self.output_directory / request.job_id
        document = extracted_document or self.extractor.extract(request.source_path, request.job_id, request.source_filename, "application/octet-stream")
        if self.generator is None:
            raise RuntimeError("A configured dataset generation provider is required.")
        if isinstance(self.generator, RunPodDatasetGenerator):
            dataset = self.generator.generate(document, request.dataset_prompt, generation_id=request.job_id, file_id=file_id or request.job_id, model=model, on_progress=on_progress)
        else:
            dataset = self.generator.generate(document, request.dataset_prompt)
        dataset, validation_report = self.phase4_validator.validate(dataset, document)
        if validation_report.status == "failed":
            raise ValueError("Generated dataset failed Phase 4 schema or source-evidence validation.")

        quality_review = None
        if self.quality_review_enabled:
            if self.quality_reviewer is None or self.quality_reviser is None:
                raise RuntimeError("Quality review is enabled but no review services are configured.")
            if on_state:
                on_state("reviewing", 84, "reviewing")
            quality_review = self.quality_reviewer.review(dataset, request.dataset_prompt, validation_report)
            if hasattr(self.generator, "last_run"):
                self.generator.last_run["review_attempts"] = quality_review.review_batch_count
                self.generator.last_run["provider_jobs"] += self.quality_reviewer.provider_jobs
            revision_available = getattr(self.generator, "last_run", {}).get("revision_attempts", 0) < self.revision_policy.max_revision_attempts
            if quality_review.blocking_issues and revision_available and self.revision_policy.allow_quality_revision:
                if on_state:
                    on_state("revising", 89, "revising")
                quality_review = quality_review.model_copy(update={"revision_attempted": True})
                source_context = "\n".join(f"SOURCE UNIT: {item.source_ref}\nTEXT: {item.quote}" for record in dataset.records for item in record.evidence)
                replacements = self.quality_reviser.revise(dataset.records, quality_review.issues, request.dataset_prompt, request.job_id, file_id or request.job_id, source_context)
                replacement_map = {record.metadata.get("record_id"): record for record in replacements}
                dataset = dataset.model_copy(update={"records": [replacement_map.get(record.metadata.get("record_id"), record) for record in dataset.records]})
                if hasattr(self.generator, "last_run"):
                    self.generator.last_run["revision_attempts"] = getattr(self.generator, "last_run", {}).get("revision_attempts", 0) + 1
                    self.generator.last_run["provider_jobs"] += self.quality_reviser.provider_jobs
                    self.generator.last_run["revision_reason"] = sorted({issue.code for issue in quality_review.issues})
                if on_state:
                    on_state("revalidating", 93, "revalidating")
                dataset, validation_report = self.phase4_validator.validate(dataset, document)
                if validation_report.status == "failed":
                    raise ValueError("Revised dataset failed the complete Phase 4 deterministic validation gate.")
                quality_review = quality_review.model_copy(update={"revision_succeeded": True, "revision_required": False})
            elif quality_review.blocking_issues:
                raise ValueError("Dataset quality review found blocking issues and the single revision budget is unavailable.")
            review_status = "passed_with_warnings" if quality_review.warnings else "passed"
            quality_review = quality_review.model_copy(update={"status": review_status})
            validation_report = validation_report.model_copy(update={"quality_review": {"status": review_status, "blockingIssues": quality_review.blocking_issues, "warnings": quality_review.warnings, "revisionAttempted": quality_review.revision_attempted, "revisionSucceeded": quality_review.revision_succeeded}})
        if on_state:
            on_state("packaging", 96, "packaging")
        extension = "json" if request.output_format.value == "json" else "csv"
        exported = self.exporter.export(dataset, request.output_format, job_directory / f"dataset.{extension}")
        manifest = GenerationManifest(job_id=request.job_id, source_file=request.source_filename, requested_format=request.output_format, record_count=len(dataset.records), model=model, generationBatchCount=getattr(self.generator, "last_run", {}).get("generation_batch_count", 1), validation_status=validation_report.status, qualityReviewStatus=("passed_with_warnings" if quality_review and quality_review.warnings else "passed" if quality_review else "not_evaluated"))
        archive = self.packager.package(exported, manifest, job_directory / "dataset.zip", validation_report=validation_report.model_dump(mode="json"), quality_review=quality_review.model_dump(mode="json", by_alias=True) if quality_review else None)
        message = "Techie custom-agentic dataset created, deterministically validated, and quality reviewed." if quality_review else "Techie custom-agentic dataset created and source-evidence validated."
        return GenerationResult(job_id=request.job_id, source_filename=request.source_filename, output_format=request.output_format, status="complete", record_count=len(dataset.records), archive_filename=archive.name, download_url=f"/api/download/{request.job_id}", message=message, validation_report=validation_report, quality_review=quality_review)
