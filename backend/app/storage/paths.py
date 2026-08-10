from pathlib import Path
from ..core.config import Settings
from .cleanup import cleanup_orphan_uploads


def prepare_storage(settings: Settings) -> None:
    settings.output_directory.mkdir(parents=True, exist_ok=True)
    settings.temp_upload_directory.mkdir(parents=True, exist_ok=True)
    cleanup_orphan_uploads(settings)
