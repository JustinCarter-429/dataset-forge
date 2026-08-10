from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from .enums import GenerationStage, OutputFormat
from .extraction_models import CanonicalExtractedDocument


class NormalizedDocument(BaseModel):
    source_filename: str
    source_type: str
    text: str


class DatasetRecord(BaseModel):
    instruction: str
    input: str
    output: str
    metadata: dict[str, str] = Field(default_factory=dict)
    context: str = ""
    expected_output: str = ""
    category: str = "general"
    difficulty: str = "medium"
    source_refs: list[str] = Field(default_factory=list)
    evidence: list["EvidenceReference"] = Field(default_factory=list)


class EvidenceReference(BaseModel):
    source_ref: str
    quote: str


class CanonicalDataset(BaseModel):
    records: list[DatasetRecord]
    placeholder: bool = False


class ValidationResult(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    code: str
    severity: str
    message: str
    record_id: str | None = None
    field: str | None = None


class GroundingSummary(BaseModel):
    status: str = "not_evaluated"
    total_records: int = 0
    grounded_records: int = 0
    ungrounded_records: int = 0
    total_evidence_items: int = 0
    verified_evidence_items: int = 0
    failed_evidence_items: int = 0
    grounding_percent: float = 0


class QualitySummary(BaseModel):
    status: str = "failed"
    schema_valid: bool = False
    total_generated_records: int = 0
    final_record_count: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    category_count: int = 0
    difficulty_distribution: dict[str, int] = Field(default_factory=dict)
    exact_duplicates_removed: int = 0
    near_duplicate_pairs: int = 0
    warnings: list[ValidationIssue] = Field(default_factory=list)


class ValidationReport(BaseModel):
    schema_version: str = "2.0"
    status: str = "failed"
    schema_valid: bool = False
    records: dict[str, object] = Field(default_factory=dict)
    grounding: GroundingSummary = Field(default_factory=GroundingSummary)
    duplicates: dict[str, object] = Field(default_factory=dict)
    quality: QualitySummary = Field(default_factory=QualitySummary)
    issues: list[ValidationIssue] = Field(default_factory=list)
    quality_review: dict[str, object] | None = Field(default=None, alias="qualityReview")

    model_config = {"populate_by_name": True}


class QualityReviewIssue(BaseModel):
    code: str
    severity: str
    record_ids: list[str] = Field(alias="recordIds")
    message: str
    suggested_action: str = Field(alias="suggestedAction")

    model_config = {"populate_by_name": True, "extra": "forbid"}


class QualityReview(BaseModel):
    review_version: str = Field(alias="reviewVersion")
    provider: str = "runpod_serverless"
    model: str
    prompt_version: str = Field(alias="promptVersion")
    review_batch_count: int = Field(alias="reviewBatchCount")
    completed_review_batches: int = Field(alias="completedReviewBatches")
    issues_found: int = Field(alias="issuesFound")
    blocking_issues: int = Field(alias="blockingIssues")
    warnings: int
    revision_required: bool = Field(alias="revisionRequired")
    revision_attempted: bool = Field(alias="revisionAttempted")
    revision_succeeded: bool = Field(alias="revisionSucceeded")
    summary: str
    issues: list[QualityReviewIssue] = Field(default_factory=list)
    status: str = "passed"

    model_config = {"populate_by_name": True, "extra": "forbid"}


class ReviewSummary(BaseModel):
    status: str = "pending"
    issue_count: int = Field(0, alias="issueCount")
    blocking_issue_count: int = Field(0, alias="blockingIssueCount")
    warning_count: int = Field(0, alias="warningCount")
    revision_attempted: bool = Field(False, alias="revisionAttempted")
    revision_succeeded: bool = Field(False, alias="revisionSucceeded")
    revision_attempts: int = Field(0, alias="revisionAttempts")
    review_attempts: int = Field(0, alias="reviewAttempts")
    provider_jobs: int = Field(0, alias="providerJobs")
    repair_reason: str | None = Field(None, alias="repairReason")
    revision_reason: list[str] = Field(default_factory=list, alias="revisionReason")

    model_config = {"populate_by_name": True}


class GenerationManifest(BaseModel):
    job_id: str
    source_file: str
    requested_format: OutputFormat
    record_count: int
    phase: str = "phase_5_runpod_quality_review"
    generator: str = "runpod_gpt_oss_20b"
    provider: str = "runpod_serverless"
    model: str = "gpt-oss-20b"
    prompt_version: str = "dataset-author-v3"
    generation_batch_count: int = Field(1, alias="generationBatchCount")
    schema_version: str = "2.0"
    validation_status: str = "failed"
    quality_review_status: str = Field("not_evaluated", alias="qualityReviewStatus")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"populate_by_name": True}


class GenerationResult(BaseModel):
    job_id: str
    source_filename: str
    output_format: OutputFormat
    status: GenerationStage
    record_count: int
    archive_filename: str
    download_url: str
    message: str
    validation_report: ValidationReport | None = Field(default=None, alias="validationReport")
    quality_review: QualityReview | None = Field(default=None, alias="qualityReview")

    model_config = {"populate_by_name": True}


class UploadedFile(BaseModel):
    id: str
    name: str
    size_bytes: int = Field(alias="sizeBytes")
    mime_type: str = Field(alias="mimeType")
    extension: str
    status: str = "uploaded"

    model_config = {"populate_by_name": True}


class GenerationRequest(BaseModel):
    file_id: str | None = Field(default=None, alias="fileId")
    file_ids: list[str] = Field(default_factory=list, alias="fileIds", max_length=3)
    dataset_prompt: str = Field(alias="datasetPrompt")
    output_format: OutputFormat = Field(alias="outputFormat")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def selected_files(self):
        ids = self.file_ids or ([self.file_id] if self.file_id else [])
        if not ids or len(ids) > 3 or len(set(ids)) != len(ids):
            raise ValueError("Select between one and three distinct source documents.")
        self.file_ids = ids
        self.file_id = ids[0]
        return self


class ValidationSummary(BaseModel):
    schema_valid: bool | None = Field(alias="schemaValid")
    total_records: int = Field(alias="totalRecords")
    valid_records: int = Field(alias="validRecords")
    invalid_records: int = Field(alias="invalidRecords")
    grounding_status: str = Field(alias="groundingStatus")
    grounded_records: int = Field(0, alias="groundedRecords")
    total_evidence_items: int = Field(0, alias="totalEvidenceItems")
    verified_evidence_items: int = Field(0, alias="verifiedEvidenceItems")
    quality_status: str = Field("failed", alias="qualityStatus")
    exact_duplicates_removed: int = Field(0, alias="exactDuplicatesRemoved")
    near_duplicate_pairs: int = Field(0, alias="nearDuplicatePairs")

    model_config = {"populate_by_name": True}


class GenerationJob(BaseModel):
    id: str
    status: str
    stage: str
    progress: dict[str, object]
    file: dict[str, str]
    output: dict[str, object]
    validation: ValidationSummary | None = None
    validation_report: ValidationReport | None = Field(None, alias="validationReport")
    quality_review: QualityReview | None = Field(None, alias="qualityReview")
    review: ReviewSummary | None = None
    package_ready: bool = Field(False, alias="packageReady")
    capabilities: dict[str, str]
    extraction: CanonicalExtractedDocument | None = None
    analysis: dict[str, object] | None = None
    provider: dict[str, object] | None = None
    batch: dict[str, int] | None = None
    error: dict[str, str] | None = None

    model_config = {"populate_by_name": True}


class PipelineInput(BaseModel):
    job_id: str
    source_path: Path
    source_filename: str
    dataset_prompt: str
    output_format: OutputFormat
