from functools import lru_cache
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)


class Settings:
    app_environment: str = os.getenv("APP_ENVIRONMENT", "development")
    frontend_origin: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
    max_upload_size: int = int(os.getenv("MAX_UPLOAD_SIZE", str(25 * 1024 * 1024)))
    output_directory: Path = Path(os.getenv("OUTPUT_DIRECTORY", "app/outputs"))
    temp_upload_directory: Path = Path(os.getenv("TEMP_UPLOAD_DIRECTORY", "app/uploads"))
    quality_validator_mode: str = os.getenv("QUALITY_VALIDATOR_MODE", "same_model")
    public_research_enabled: bool = os.getenv("PUBLIC_RESEARCH_ENABLED", "false").lower() == "true"
    lm_studio_enabled: bool = os.getenv("LM_STUDIO_ENABLED", "false").lower() == "true"
    lm_studio_base_url: str = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234")
    lm_studio_model: str = os.getenv("LM_STUDIO_MODEL", "openai/gpt-oss-20b")
    lm_studio_timeout_seconds: float = float(os.getenv("LM_STUDIO_TIMEOUT_SECONDS", "45"))
    carter_knowledge_database: Path = Path(os.getenv("CARTER_KNOWLEDGE_DATABASE", "runtime/carter-knowledge.sqlite3"))
    app_test_mode: bool = os.getenv("APP_ENVIRONMENT", "development") == "test" and os.getenv("CARTER_TEST_PROVIDER", "") == "deterministic"


@lru_cache
def get_settings() -> Settings:
    return Settings()
