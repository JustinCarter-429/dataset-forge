import pytest
from pydantic import ValidationError

from app.domain.models import GenerationRequest
from app.providers.deterministic import enabled


def test_generation_request_rejects_four_documents_before_dispatch():
    with pytest.raises(ValidationError, match="one and three"):
        GenerationRequest(fileId="one", fileIds=["one", "two", "three", "four"], datasetPrompt="Create records", outputFormat="json")


def test_deterministic_provider_is_disabled_without_explicit_test_mode(monkeypatch):
    monkeypatch.delenv("APP_ENVIRONMENT", raising=False)
    monkeypatch.delenv("CARTER_TEST_PROVIDER", raising=False)
    assert enabled() is False


def test_deterministic_provider_requires_test_environment(monkeypatch):
    monkeypatch.setenv("APP_ENVIRONMENT", "development")
    monkeypatch.setenv("CARTER_TEST_PROVIDER", "deterministic")
    assert enabled() is False
    monkeypatch.setenv("APP_ENVIRONMENT", "test")
    assert enabled() is True
