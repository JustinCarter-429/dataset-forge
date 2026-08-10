import re
from pathlib import Path

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def sanitize_filename(filename: str) -> str:
    name = Path(filename or "document").name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "document"


def validate_filename(filename: str) -> str:
    safe_name = sanitize_filename(filename)
    if Path(safe_name).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return safe_name


def safe_download_name(source_name: str) -> str:
    stem = Path(source_name or "dataset").stem
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-_") or "dataset"
    return f"{stem}-dataset.zip"
