import logging
import uuid
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from ...core.config import get_settings
from ...domain.enums import OutputFormat, GenerationStage
from ...domain.models import GenerationJob, GenerationRequest, GenerationResult, PipelineInput, UploadedFile, ValidationSummary, ReviewSummary
from ...domain.extraction_models import CanonicalExtractedDocument
from ...services.job_store import StoredFile, job_store
from ...services.extraction import ExtractionError, ExtractionService
from ...services.extraction_analysis import analyze_extraction
from ...services.pipeline import PipelineService
from ...providers.config import provider_config_from_env
from ...providers.runpod import RunPodProvider
from ...providers.contracts import ProviderError
from ...providers.deterministic import DeterministicCarterProvider, enabled as deterministic_enabled
from ...services.quality_review import QualityReviewService, QualityRevisionService
from ...services.carter import LMStudioCarterProvider, RunPodCarterProvider
from ...carter.runtime import CarterPromptPackage, CarterPromptPackageError
from ...carter.production import CarterDatasetGenerationService
from ...carter.dynamic_dataset import export_canonical_csv, export_canonical_json, quality_gate
from ...services.packaging import ZipDatasetPackager
from ...domain.models import GenerationManifest
from ...utils.files import validate_filename

logger = logging.getLogger(__name__)
router = APIRouter()
_provider_factory = lambda config: RunPodProvider(config)
_active_providers: dict[str, RunPodProvider] = {}

def _safe_error(code: str, message: str): return {"code": code, "message": message}


def _provider_diagnostics(provider) -> dict[str, object]:
    metrics = getattr(provider, "metrics", {})
    return {key: int(value) for key, value in metrics.items() if key in {"providerSubmitAttempts", "providerJobsCreated", "providerJobsCompleted", "providerJobsFailed", "providerStatusPolls", "providerTransportRetries", "providerCancelCalls"}}


def _combined_document(request: GenerationRequest) -> tuple[list[StoredFile], CanonicalExtractedDocument]:
    stored_files = [job_store.get_file(file_id) for file_id in request.file_ids]
    if any(item is None for item in stored_files):
        raise HTTPException(404, "One or more uploaded source files were not found.")
    records = [item for item in stored_files if item is not None]
    extracted: list[CanonicalExtractedDocument] = []
    for stored in records:
        document = stored.extraction or ExtractionService().extract(stored.path, stored.record.id, stored.record.name, stored.record.mime_type)
        job_store.update_file(stored.record.id, extraction=document, record=stored.record.model_copy(update={"status": "ready"}))
        extracted.append(document)
    if len(extracted) == 1: return records, extracted[0]
    # Element IDs remain their original globally unique extraction IDs; only the
    # outer document is a multi-document generation container.
    first = extracted[0]
    combined = first.model_copy(update={"document_id": "multi-" + uuid.uuid4().hex, "source_file_id": "multi", "source_filename": ", ".join(doc.source_filename for doc in extracted), "elements": [element for doc in extracted for element in doc.elements]})
    return records, combined


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
    stored = job_store.get_file(request.file_id or "")
    if not stored: return
    settings = get_settings()
    def stage(name: str, percent: int, current_stage: str | None = None):
        job_store.update_job(job_id, status=name, stage=name, progress={"percent": percent, "currentStage": current_stage or name})
    try:
        if job_store.is_cancelled(job_id): return
        stored.record = stored.record.model_copy(update={"status": "extracting"})
        stage("extracting", 15)
        stored_files, extracted = _combined_document(request)
        stage("analyzing", 30)
        analysis = analyze_extraction(extracted)
        job_store.update_job(job_id, extraction=extracted, analysis=analysis)
        job_store.update_file(request.file_id or "", record=stored.record.model_copy(update={"status": "ready"}))
        stage("planning", 35, "planning")
        config = provider_config_from_env()
        runtime = request.runtime
        if runtime not in {"runpod", "local_lm_studio"}: raise ValueError("Unsupported Carter runtime.")
        if deterministic_enabled(): provider = DeterministicCarterProvider(runtime)
        elif runtime == "runpod": provider = RunPodCarterProvider(_provider_factory(config))
        else: provider = LMStudioCarterProvider(settings.lm_studio_base_url, settings.lm_studio_model, settings.lm_studio_timeout_seconds, settings.lm_studio_enabled, settings.carter_max_tokens)
        _active_providers[job_id] = getattr(provider, "provider", provider)
        availability = provider.available()
        if not availability.get("available"): raise ProviderError("RUNTIME_UNAVAILABLE", "The selected Carter runtime is unavailable.")
        job_store.update_job(job_id, runtime=runtime, provider={"name": runtime, "model": availability.get("model"), "state": "configured"})
        phase_percent = {"planning": 40, "generating": 55, "tool_use": 62, "reviewing": 76, "revising": 84}
        def on_phase(phase: str):
            job_store.update_job(job_id, status=phase, stage=phase, progress={"percent": phase_percent[phase], "currentStage": phase}, provider={"name": runtime, "model": availability.get("model"), "state": "running"})
        documents = [item.extraction for item in stored_files if item.extraction]
        package = CarterPromptPackage.load()
        result = CarterDatasetGenerationService(package, provider, knowledge_path=settings.carter_knowledge_database.parent / f"{job_id}.sqlite3", on_phase=on_phase, cancelled=lambda: job_store.is_cancelled(job_id)).generate(runtime=runtime, user_request=request.dataset_prompt.strip(), output_format=request.output_format.value, documents=documents)
        if job_store.is_cancelled(job_id): return
        stage("validating", 90)
        allowed_refs = {element.element_id for document in documents for element in document.elements if element.text.strip()}
        export_dataset, quality = quality_gate(package, result.dataset, allowed_source_refs=allowed_refs)
        if not quality.export_eligible:
            raise CarterPromptPackageError("No records passed the deterministic quality gate.")
        archive = settings.output_directory / job_id / "dataset.zip"
        output = settings.output_directory / job_id / ("dataset.json" if request.output_format.value == "json" else "dataset.csv")
        (export_canonical_json if request.output_format.value == "json" else export_canonical_csv)(export_dataset, output)
        quality_payload = quality.as_dict()
        manifest = GenerationManifest(job_id=job_id, source_file=extracted.source_filename, requested_format=request.output_format, record_count=len(export_dataset.records), phase="carter_1_0", generator="carter_1_0", provider=runtime, model=str(availability.get("model") or "unknown"), prompt_version=package.package_version, schema_version="carter-1.0", validation_status=quality_payload["status"], quality_review_status=result.review["recommendation"])
        ZipDatasetPackager().package(output, manifest, archive, validation_report=quality_payload, quality_review=result.review)
        validation = ValidationSummary(schemaValid=quality.schema_failures == 0, totalRecords=quality.total_records, validRecords=quality.accepted_records, invalidRecords=quality.rejected_records + quality.quarantined_records, groundingStatus="passed" if quality.grounding_failures == 0 else "failed", groundedRecords=quality.accepted_records, totalEvidenceItems=sum(len(record["evidence"]) for record in export_dataset.records), verifiedEvidenceItems=sum(len(record["evidence"]) for record in export_dataset.records), qualityStatus="passed" if not quality.findings else "passed_with_warnings", exactDuplicatesRemoved=quality.duplicate_records)
        stage("packaging", 96)
        review_summary = ReviewSummary(status=result.review["recommendation"], issueCount=len(result.review["issues"]), blockingIssueCount=sum(issue["severity"] == "major" for issue in result.review["issues"]), warningCount=sum(issue["severity"] == "warning" for issue in result.review["issues"]), revisionAttempted=bool(result.revisions), revisionSucceeded=bool(result.revisions), revisionAttempts=result.revisions, reviewAttempts=result.calls["review"], providerJobs=sum(result.calls.values()))
        job_store.update_job(job_id, status="completed", stage="completed", progress={"percent": 100, "currentStage": "completed"}, output={"requestedFormat": request.output_format.value, "recordCount": len(export_dataset.records), "finalRecordCount": len(export_dataset.records), "sizeBytes": archive.stat().st_size, "qualitySummary": quality_payload}, validation=validation, review=review_summary, package_ready=True, provider={"name": runtime, "model": availability.get("model"), "state": "completed", "carterCalls": result.calls, "tools": result.tools_executed}, capabilities={"extraction": "docling_pdf_docx_or_plain_text", "generation": "carter_1_0", "groundingValidation": "carter_dynamic_evidence", "qualityReview": "carter_bounded_review"})
    except ExtractionError as exc:
        logger.warning("Extraction failed for job %s: %s", job_id, exc.code)
        job_store.update_file(request.file_id or "", record=stored.record.model_copy(update={"status": "failed"}))
        job_store.update_job(job_id, status="failed", stage="extracting", error=_safe_error(exc.code, exc.message))
    except ProviderError as exc:
        logger.warning("Provider failed for job %s: %s", job_id, exc.code)
        job_store.update_file(request.file_id or "", record=stored.record.model_copy(update={"status": "failed"}))
        review_code = exc.code.startswith("QUALITY_REVIEW") or exc.code.startswith("QUALITY_REVISION")
        if job_store.is_cancelled(job_id): return
        job_store.update_job(job_id, status="failed", stage="validating" if review_code else "generating", progress={"percent": 86 if review_code else 35, "currentStage": "reviewing" if review_code else "generating"}, provider={"name": "runpod_serverless", "state": "failed", **_provider_diagnostics(provider)}, error=_safe_error(exc.code, "Dataset quality review could not be completed safely." if review_code else exc.message))
    except ValueError as exc:
        logger.warning("Validation failed for job %s: %s", job_id, str(exc))
        job_store.update_file(request.file_id or "", record=stored.record.model_copy(update={"status": "failed"}))
        job_store.update_job(job_id, status="failed", stage="validating", progress={"percent": 80, "currentStage": "validating"}, error=_safe_error("VALIDATION_FAILED", "Some generated records could not be verified against the uploaded source."))
    except Exception:
        logger.exception("Generation failed for job %s", job_id)
        job_store.update_file(request.file_id or "", record=stored.record.model_copy(update={"status": "failed"}))
        if not job_store.is_cancelled(job_id): job_store.update_job(job_id, status="failed", stage="generating", error=_safe_error("GENERATION_FAILED", "Dataset generation failed. Your inputs have been preserved."))
    finally:
        _active_providers.pop(job_id, None)
        job_store.release_generation()
        if job_store.get_job(job_id) and job_store.get_job(job_id).status in {"completed", "failed", "cancelled"}:
            for selected_id in request.file_ids:
                selected = job_store.get_file(selected_id)
                if selected:
                    selected.path.unlink(missing_ok=True)


@router.post("/generations", status_code=202)
def create_generation(request: GenerationRequest, background_tasks: BackgroundTasks):
    stored = job_store.get_file(request.file_id or "")
    if not stored: raise HTTPException(404, "Uploaded file not found.")
    if not request.dataset_prompt.strip(): raise HTTPException(422, "Please describe the dataset you want to create.")
    if not job_store.try_acquire_generation():
        raise HTTPException(409, "Another dataset generation is already in progress. Please wait for it to finish.")
    job_id = uuid.uuid4().hex
    if request.runtime not in {"runpod", "local_lm_studio"}: raise HTTPException(422, "Unsupported Carter runtime.")
    job = GenerationJob(id=job_id, status="queued", stage="queued", progress={"percent": 3, "currentStage": "queued"}, file={"id": stored.record.id, "name": stored.record.name}, output={"requestedFormat": request.output_format.value, "recordCount": None, "finalRecordCount": None, "sizeBytes": None}, capabilities={"extraction": "docling_pdf_docx_or_plain_text", "generation": "carter_1_0", "groundingValidation": "carter_dynamic_evidence", "qualityReview": "carter_bounded_review"}, provider={"name": request.runtime, "state": "not_started"}, runtime=request.runtime)
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
