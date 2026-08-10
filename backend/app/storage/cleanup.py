from datetime import datetime, timedelta, timezone
from pathlib import Path
from ..core.config import Settings


def cleanup_orphan_uploads(settings: Settings, *, max_age_hours: int = 24) -> int:
    """Remove only stale upload fragments; completed output packages are retained."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    removed = 0
    directory = settings.temp_upload_directory
    if not directory.exists():
        return 0
    for path in directory.iterdir():
        if not path.is_file():
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        if modified < cutoff:
            path.unlink(missing_ok=True)
            removed += 1
    return removed
