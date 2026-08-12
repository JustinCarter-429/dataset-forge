from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from datetime import datetime, timezone
from ..domain.models import GenerationJob, UploadedFile
from ..domain.extraction_models import CanonicalExtractedDocument


@dataclass
class StoredFile:
    record: UploadedFile
    path: Path
    extraction: CanonicalExtractedDocument | None = None


class InMemoryJobStore:
    """Process-local Phase 1A store; durable jobs arrive with a later phase."""
    def __init__(self):
        self._files: dict[str, StoredFile] = {}
        self._jobs: dict[str, GenerationJob] = {}
        self._lock = Lock()
        self._cancel_requested: set[str] = set()
        self._active_jobs = 0

    def add_file(self, stored: StoredFile) -> None:
        with self._lock: self._files[stored.record.id] = stored

    def get_file(self, file_id: str) -> StoredFile | None:
        with self._lock: return self._files.get(file_id)

    def update_file(self, file_id: str, **changes) -> StoredFile | None:
        with self._lock:
            current = self._files.get(file_id)
            if not current: return None
            if "record" in changes: current.record = changes["record"]
            if "extraction" in changes: current.extraction = changes["extraction"]
            return current

    def add_job(self, job: GenerationJob) -> None:
        with self._lock: self._jobs[job.id] = job

    def try_acquire_generation(self) -> bool:
        with self._lock:
            if self._active_jobs >= 1:
                return False
            self._active_jobs += 1
            return True

    def release_generation(self) -> None:
        with self._lock: self._active_jobs = max(0, self._active_jobs - 1)

    def request_cancel(self, job_id: str) -> GenerationJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if job.status in {"completed", "failed", "cancelled"}:
                return job
            self._cancel_requested.add(job_id)
            updated = job.model_copy(update={"status": "cancelled", "stage": "cancelled", "progress": {"percent": job.progress.get("percent", 0), "currentStage": "cancelled"}, "package_ready": False, "error": {"code": "GENERATION_CANCELLED", "message": "Generation cancelled. No package was created."}})
            self._jobs[job_id] = updated
            return updated

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock: return job_id in self._cancel_requested

    def get_job(self, job_id: str) -> GenerationJob | None:
        with self._lock: return self._jobs.get(job_id)

    def update_job(self, job_id: str, **changes) -> GenerationJob | None:
        with self._lock:
            current = self._jobs.get(job_id)
            if not current: return None
            if job_id in self._cancel_requested and changes.get("status") not in {"cancelled"}:
                return current
            now = datetime.now(timezone.utc)
            changes.setdefault("updated_at", now)
            if any(key in changes for key in ("status", "stage", "progress", "batch")):
                changes.setdefault("last_progress_at", now)
            if "stage" in changes and changes["stage"] != current.stage:
                changes.setdefault("current_stage_started_at", now)
            updated = current.model_copy(update=changes)
            self._jobs[job_id] = updated
            return updated


job_store = InMemoryJobStore()
