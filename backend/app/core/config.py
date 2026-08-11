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
    # gpt-oss-20b can cold-start slowly locally; this remains bounded and is
    # owned by the transport rather than an arbitrary UI polling deadline.
    lm_studio_timeout_seconds: float = float(os.getenv("LM_STUDIO_TIMEOUT_SECONDS", "180"))
    carter_max_tokens: int = int(os.getenv("CARTER_MAX_TOKENS", "4096"))
    carter_generation_batch_size: int = int(os.getenv("CARTER_GENERATION_BATCH_SIZE", "5"))
    carter_generation_no_content_retries: int = int(os.getenv("CARTER_GENERATION_NO_CONTENT_RETRIES", "2"))
    carter_knowledge_database: Path = Path(os.getenv("CARTER_KNOWLEDGE_DATABASE", "runtime/carter-knowledge.sqlite3"))
    app_test_mode: bool = os.getenv("APP_ENVIRONMENT", "development") == "test" and os.getenv("CARTER_TEST_PROVIDER", "") == "deterministic"

    def __init__(self):
        if not 1 <= self.carter_generation_batch_size <= 20:
            raise ValueError("CARTER_GENERATION_BATCH_SIZE must be between 1 and 20.")
        if not 0 <= self.carter_generation_no_content_retries <= 2:
            raise ValueError("CARTER_GENERATION_NO_CONTENT_RETRIES must be between 0 and 2.")


@lru_cache
def get_settings() -> Settings:
    return Settings()
