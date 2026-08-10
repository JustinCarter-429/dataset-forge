from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...core.config import get_settings
from ...providers.config import provider_config_from_env
from ...providers.runpod import RunPodProvider
from ...providers.contracts import ProviderError
from ...services.carter import CarterAskService, KnowledgeStore, LMStudioCarterProvider, RunPodCarterProvider
from ...services.extraction import ExtractionError, ExtractionService
from ...services.job_store import job_store

router = APIRouter(prefix="/carter", tags=["carter"])

class AskRequest(BaseModel):
    question: str = Field(max_length=4000)
    runtime: Literal["cloud", "local"]
    document_ids: list[str] = Field(default_factory=list, alias="documentIds", max_length=3)
    model_config = {"populate_by_name": True}

class IngestRequest(BaseModel):
    file_ids: list[str] = Field(alias="fileIds", min_length=1, max_length=3)
    model_config = {"populate_by_name": True}

def _store() -> KnowledgeStore: return KnowledgeStore(get_settings().carter_knowledge_database)

def _local() -> LMStudioCarterProvider:
    settings = get_settings(); return LMStudioCarterProvider(settings.lm_studio_base_url, settings.lm_studio_model, settings.lm_studio_timeout_seconds, settings.lm_studio_enabled)

@router.get("/runtimes")
def runtimes():
    local = _local().available()
    try:
        provider_config_from_env().validate(); cloud = {"configured": True, "available": True}
    except ProviderError:
        cloud = {"configured": False, "available": False}
    return {"assistant":"Carter 1.0", "carterVersion":"1.0", "cloud":cloud, "local":local}

@router.post("/ingest")
def ingest(request: IngestRequest):
    store = _store(); extractor = ExtractionService(); added = []
    for file_id in request.file_ids:
        stored = job_store.get_file(file_id)
        if not stored: raise HTTPException(404, "Source file not found.")
        try:
            extracted = stored.extraction or extractor.extract(stored.path, stored.record.id, stored.record.name, stored.record.mime_type)
            job_store.update_file(file_id, extraction=extracted); store.ingest(extracted); added.append({"documentId":extracted.document_id,"name":extracted.source_filename})
        except ExtractionError as exc: raise HTTPException(422, exc.message) from exc
        except ValueError as exc: raise HTTPException(422, str(exc)) from exc
    return {"documents": added}

@router.get("/documents")
def documents(): return {"documents": _store().documents(), "limit": 3}

@router.post("/ask")
def ask(request: AskRequest):
    try:
        if request.runtime == "local": provider = _local()
        else: provider = RunPodCarterProvider(RunPodProvider(provider_config_from_env()))
        return CarterAskService(_store(), provider).ask(request.question, request.document_ids)
    except ProviderError as exc: raise HTTPException(503, {"code":exc.code, "message":exc.message}) from exc
    except ValueError as exc: raise HTTPException(422, str(exc)) from exc
