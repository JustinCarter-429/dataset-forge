import logging
import uuid
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from ...core.config import get_settings
from ...domain.enums import OutputFormat, GenerationStage
from ...domain.models import GenerationJob, GenerationRequest, GenerationResult, PipelineInput, UploadedFile, ValidationSummary, ReviewSummary
from ...services.job_store import StoredFile, job_store
from ...services.extraction import ExtractionError, ExtractionService
from ...services.extraction_analysis import analyze_extraction
from ...services.pipeline import PipelineService
from ...services.generation import RunPodDatasetGenerator
from ...providers.config import provider_config_from_env
from ...providers.runpod import RunPodProvider
from ...providers.contracts import ProviderError
from ...services.quality_review import QualityReviewService, QualityRevisionService
from ...utils.files import validate_filename

logger = logging.getLogger(__name__)
router = APIRouter()
_provider_factory = lambda config: RunPodProvider(config)
_active_providers: dict[str, RunPodProvider] = {}

def _safe_error(code: str, message: str): return {"code": code, "message": message}


def _provider_diagnostics(provider) -> dict[str, object]:
    metrics = getattr(provider, "metrics", {})
    return {key: int(value) for key, value in metrics.items() if key in {"providerSubmitAttempts", "providerJobsCreated", "providerJobsCompleted", "providerJobsFailed", "providerStatusPolls", "providerTransportRetries", "providerCancelCalls"}}


async def _save_upload(file: UploadFile) -> UploadedFile:
    settings = get_settings()
    provider = None
    try: safe_name = validate_filename(file.filename or "")
    except ValueError as exc: raise HTTPException(422, str(exc)) from exc
    file_id = uuid.uuid4().hex
    temp_path = settings.temp_upload_directory / f"{file_id}_{safe_name}"
    size = 0
    try:
        with temp_path.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_size: raise HTTPException(413, "File is too large. The maximum size is 25 MB.")
                target.write(chunk)
        if size == 0: raise HTTPException(422, "The uploaded file is empty.")
        record = UploadedFile(id=file_id, name=safe_name, sizeBytes=size, mimeType=file.content_type or "application/octet-stream", extension=safe_name.rsplit(".", 1)[-1].lower())
        job_store.add_file(StoredFile(record, temp_path))
        return record
    except HTTPException: temp_path.unlink(missing_ok=True); raise
    finally: await file.close()


@router.post("/files", status_code=201)
async def upload_file(file: UploadFile = File(...)):
    return {"file": (await _save_upload(file)).model_dump(by_alias=True)}


@router.get("/files/{file_id}")
def get_file(file_id: str):
    stored = job_store.get_file(file_id)
    if not stored: raise HTTPException(404, "File not found.")
    return {"file": stored.record.model_dump(by_alias=True)}


def _run_job(job_id: str, request: GenerationRequest):
    stored = job_store.get_file(request.file_id)
    if not stored: return
    settings = get_settings()
    def stage(name: str, percent: int, current_stage: str | None = None):
        job_store.update_job(job_id, status=name, stage=name, progress={"percent": percent, "currentStage": current_stage or name})
    try:
        if job_store.is_cancelled(job_id): return
        stored.record = stored.record.model_copy(update={"status": "extracting"})
        stage("extracting", 15)
        extractor = ExtractionService()
        extracted = extractor.extract(stored.path, stored.record.id, stored.record.name, stored.record.mime_type)
        job_store.update_file(request.file_id, extraction=extracted, record=stored.record.model_copy(update={"status": "analyzing"}))
        stage("analyzing", 30)
        analysis = analyze_extraction(extracted)
        job_store.update_job(job_id, extraction=extracted, analysis=analysis)
        job_store.update_file(request.file_id, record=stored.record.model_copy(update={"status": "ready"}))
        stage("generating", 35, "waiting_for_ai_worker")
        config = provider_config_from_env()
        provider = _provider_factory(config)
        _active_providers[job_id] = provider
        provider.cancel_check = lambda: job_store.is_cancelled(job_id)
        provider.on_job_created = lambda external_id: job_store.update_job(job_id, provider={"name": "runpod_serverless", "model": config.model, "state": "queued", "externalJobId": external_id})
        job_store.update_job(job_id, provider={"name": "runpod_serverless", "model": config.model, "state": "configured"})
        generator = RunPodDatasetGenerator(provider, max_model_len=config.max_model_len, records_per_batch=config.records_per_batch, max_dataset_records=config.max_dataset_records)
        def generation_progress(percent: int, detail: str):
            current = job_store.get_job(job_id)
            if job_store.is_cancelled(job_id): return
            batch = current.batch if current else None
            if detail.startswith("batch "):
                import re
                match = re.search(r"batch (\d+) of (\d+)", detail)
                if match: batch = {"completed": int(match.group(1)) - 1, "total": int(match.group(2))}
            job_store.update_job(job_id, status="generating", stage="generating", progress={"percent": percent, "currentStage": detail}, batch=batch, provider={"name": "runpod_serverless", "model": config.model, "state": "running"})
        review_enabled = isinstance(provider, RunPodProvider) and settings.quality_validator_mode != "disabled"
        reviewer = QualityReviewService(provider, model=config.model, max_model_len=config.max_model_len) if review_enabled else None
        reviser = QualityRevisionService(provider, model=config.model) if review_enabled else None
        def agentic_state(state: str, percent: int, current_stage: str):
            job_store.update_job(job_id, status="validating", stage="validating", progress={"percent": percent, "currentStage": current_stage})
        result = PipelineService(settings.output_directory, generator=generator, quality_review_enabled=review_enabled, quality_reviewer=reviewer, quality_reviser=reviser).run(PipelineInput(job_id=job_id, source_path=stored.path, source_filename=stored.record.name, dataset_prompt=request.dataset_prompt.strip(), output_format=request.output_format), extracted_document=extracted, file_id=stored.record.id, model=config.model, on_progress=generation_progress, on_state=agentic_state)
        if job_store.is_cancelled(job_id): return
        stage("validating", 80)
        archive = settings.output_directory / job_id / "dataset.zip"
        report = result.validation_report
        grounding = report.grounding if report else None
        quality = report.quality if report else None
        validation = ValidationSummary(schemaValid=bool(report and report.schema_valid), totalRecords=result.record_count, validRecords=quality.valid_records if quality else result.record_count, invalidRecords=quality.invalid_records if quality else 0, groundingStatus=grounding.status if grounding else "failed", groundedRecords=grounding.grounded_records if grounding else 0, totalEvidenceItems=grounding.total_evidence_items if grounding else 0, verifiedEvidenceItems=grounding.verified_evidence_items if grounding else 0, qualityStatus=quality.status if quality else "failed", exactDuplicatesRemoved=quality.exact_duplicates_removed if quality else 0, nearDuplicatePairs=quality.near_duplicate_pairs if quality else 0)
        stage("packaging", 96)
        review = result.quality_review
        review_summary = ReviewSummary(status=review.status if review else "not_evaluated", issueCount=review.issues_found if review else 0, blockingIssueCount=review.blocking_issues if review else 0, warningCount=review.warnings if review else 0, revisionAttempted=review.revision_attempted if review else False, revisionSucceeded=review.revision_succeeded if review else False, revisionAttempts=generator.last_run.get("revision_attempts", 0), reviewAttempts=generator.last_run.get("review_attempts", 0), providerJobs=generator.last_run.get("provider_jobs", 0), repairReason=generator.last_run.get("repair_reason"), revisionReason=generator.last_run.get("revision_reason", []))
        job_store.update_job(job_id, status="completed", stage="completed", progress={"percent": 100, "currentStage": "completed"}, output={"requestedFormat": request.output_format.value, "recordCount": result.record_count, "finalRecordCount": result.record_count, "sizeBytes": archive.stat().st_size}, validation=validation, validation_report=report, quality_review=review, review=review_summary, package_ready=True, provider={"name": "runpod_serverless", "model": config.model, "state": "completed", **_provider_diagnostics(provider)}, capabilities={"extraction": "docling_pdf_docx_or_plain_text", "generation": "runpod_serverless_gpt_oss_20b", "groundingValidation": "phase4_deterministic_source_evidence", "qualityReview": "bounded_same_model_advisory_review"})
    except ExtractionError as exc:
        logger.warning("Extraction failed for job %s: %s", job_id, exc.code)
        job_store.update_file(request.file_id, record=stored.record.model_copy(update={"status": "failed"}))
        job_store.update_job(job_id, status="failed", stage="extracting", error=_safe_error(exc.code, exc.message))
    except ProviderError as exc:
        logger.warning("Provider failed for job %s: %s", job_id, exc.code)
        job_store.update_file(request.file_id, record=stored.record.model_copy(update={"status": "failed"}))
        review_code = exc.code.startswith("QUALITY_REVIEW") or exc.code.startswith("QUALITY_REVISION")
        if job_store.is_cancelled(job_id): return
        job_store.update_job(job_id, status="failed", stage="validating" if review_code else "generating", progress={"percent": 86 if review_code else 35, "currentStage": "reviewing" if review_code else "generating"}, provider={"name": "runpod_serverless", "state": "failed", **_provider_diagnostics(provider)}, error=_safe_error(exc.code, "Dataset quality review could not be completed safely." if review_code else exc.message))
    except ValueError as exc:
        logger.warning("Validation failed for job %s: %s", job_id, str(exc))
        job_store.update_file(request.file_id, record=stored.record.model_copy(update={"status": "failed"}))
        job_store.update_job(job_id, status="failed", stage="validating", progress={"percent": 80, "currentStage": "validating"}, error=_safe_error("VALIDATION_FAILED", "Some generated records could not be verified against the uploaded source."))
    except Exception:
        logger.exception("Generation failed for job %s", job_id)
        job_store.update_file(request.file_id, record=stored.record.model_copy(update={"status": "failed"}))
        if not job_store.is_cancelled(job_id): job_store.update_job(job_id, status="failed", stage="generating", error=_safe_error("GENERATION_FAILED", "Dataset generation failed. Your inputs have been preserved."))
    finally:
        _active_providers.pop(job_id, None)
        job_store.release_generation()
        stored = job_store.get_file(request.file_id)
        if stored and job_store.get_job(job_id) and job_store.get_job(job_id).status in {"completed", "failed", "cancelled"}:
            stored.path.unlink(missing_ok=True)


@router.post("/generations", status_code=202)
def create_generation(request: GenerationRequest, background_tasks: BackgroundTasks):
    stored = job_store.get_file(request.file_id)
    if not stored: raise HTTPException(404, "Uploaded file not found.")
    if not request.dataset_prompt.strip(): raise HTTPException(422, "Please describe the dataset you want to create.")
    if not job_store.try_acquire_generation():
        raise HTTPException(409, "Another dataset generation is already in progress. Please wait for it to finish.")
    job_id = uuid.uuid4().hex
    job = GenerationJob(id=job_id, status="queued", stage="queued", progress={"percent": 3, "currentStage": "queued"}, file={"id": stored.record.id, "name": stored.record.name}, output={"requestedFormat": request.output_format.value, "recordCount": None, "finalRecordCount": None, "sizeBytes": None}, capabilities={"extraction": "docling_pdf_docx_or_plain_text", "generation": "runpod_serverless_gpt_oss_20b", "groundingValidation": "phase4_deterministic_source_evidence", "qualityReview": "bounded_same_model_advisory_review"}, provider={"name": "runpod_serverless", "state": "not_started"})
    job_store.add_job(job); background_tasks.add_task(_run_job, job_id, request)
    return {"generationId": job_id, "status": "queued"}


@router.post("/generations/{generation_id}/cancel")
def cancel_generation(generation_id: str):
    if not generation_id.isalnum() or len(generation_id) != 32:
        raise HTTPException(404, "Generation not found.")
    current = job_store.get_job(generation_id)
    if not current: raise HTTPException(404, "Generation not found.")
    if current.status in {"completed", "failed", "cancelled"}:
        return current.model_dump(by_alias=True)
    first_request = not job_store.is_cancelled(generation_id)
    job = job_store.request_cancel(generation_id)
    provider = _active_providers.get(generation_id)
    external_id = (current.provider or {}).get("externalJobId") if current.provider else None
    if first_request and provider and external_id:
        try: provider.cancel(str(external_id))
        except ProviderError: logger.warning("Provider cancellation failed for job %s", generation_id)
    return (job or current).model_dump(by_alias=True)


@router.get("/generations/{generation_id}")
def get_generation(generation_id: str):
    if not generation_id.isalnum() or len(generation_id) != 32: raise HTTPException(404, "Generation not found.")
    job = job_store.get_job(generation_id)
    if not job: raise HTTPException(404, "Generation not found.")
    return job.model_dump(by_alias=True)


@router.get("/generations/{generation_id}/download")
def download_generation(generation_id: str):
    job = job_store.get_job(generation_id)
    if not job or not job.package_ready: raise HTTPException(409, "Dataset package is not ready yet.")
    archive = get_settings().output_directory / generation_id / "dataset.zip"
    from ...utils.files import safe_download_name
    return FileResponse(archive, media_type="application/zip", filename=safe_download_name(job.file.get("name", "dataset")))


@router.post("/generate", response_model=GenerationResult)
async def generate(file: UploadFile = File(...), dataset_prompt: str = Form(...), output_format: OutputFormat = Form(...)):
    uploaded = await _save_upload(file)
    try:
        request = GenerationRequest(fileId=uploaded.id, datasetPrompt=dataset_prompt, outputFormat=output_format)
        job_id = uuid.uuid4().hex
        job_store.add_job(GenerationJob(id=job_id, status="queued", stage="queued", progress={"percent": 3, "currentStage": "queued"}, file={"id": uploaded.id, "name": uploaded.name}, output={"requestedFormat": output_format.value, "recordCount": None, "finalRecordCount": None, "sizeBytes": None}, capabilities={"extraction": "docling_pdf_docx_or_plain_text", "generation": "runpod_serverless_gpt_oss_20b", "groundingValidation": "phase4_deterministic_source_evidence"}, provider={"name": "runpod_serverless", "state": "not_started"}))
        _run_job(job_id, request)
        job = job_store.get_job(job_id)
        if not job or not job.package_ready: raise HTTPException(500, "Generation failed. Please try again.")
        archive = get_settings().output_directory / job_id / "dataset.zip"
        return GenerationResult(job_id=job_id, source_filename=uploaded.name, output_format=output_format, status=GenerationStage.COMPLETE, record_count=job.output["recordCount"], archive_filename=archive.name, download_url=f"/api/download/{job_id}", message="Phase 1A placeholder dataset created successfully.")
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Generation failed for job %s", job_id)
        raise HTTPException(500, "Generation failed. Please try again.") from exc
    finally:
        stored = job_store.get_file(uploaded.id)
        if stored: stored.path.unlink(missing_ok=True)


@router.get("/download/{job_id}")
def download(job_id: str):
    settings = get_settings()
    if not job_id.isalnum() or len(job_id) != 32:
        raise HTTPException(404, "Dataset archive not found.")
    archive = settings.output_directory / job_id / "dataset.zip"
    if not archive.is_file():
        raise HTTPException(404, "Dataset archive not found.")
    job = job_store.get_job(job_id)
    from ...utils.files import safe_download_name
    return FileResponse(archive, media_type="application/zip", filename=safe_download_name(job.file.get("name", "dataset") if job else "dataset"))
