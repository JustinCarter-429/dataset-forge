import json
from pathlib import Path

import pytest

from app.domain.enums import OutputFormat
from app.domain.extraction_models import CanonicalExtractedDocument, ExtractionElement, ExtractionElementType, ExtractionStatistics, ExtractionValidation
from app.domain.models import CanonicalDataset, DatasetRecord, EvidenceReference, PipelineInput, ValidationReport
from app.providers.contracts import ProviderJob, ProviderError
from app.services.generation import RunPodDatasetGenerator
from app.services.pipeline import PipelineService
from app.services.quality_review import QualityReviewService, _parse_review_output


def source() -> CanonicalExtractedDocument:
    text = "Functional testing checks feature workflows and expected functionality."
    block = ExtractionElement(elementId="s1", type=ExtractionElementType.PARAGRAPH, text=text, order=1)
    return CanonicalExtractedDocument(documentId="doc", sourceFileId="file", sourceFilename="source.txt", mimeType="text/plain", extractor="plain_text", extractorVersion="1", blocks=[block], statistics=ExtractionStatistics(pageCount=None, characterCount=len(text), wordCount=9, elementCount=1, headingCount=0, paragraphCount=1, listItemCount=0, tableCount=0, tableRowCount=0, nonEmptyElementCount=1), validation=ExtractionValidation(valid=True, quality="good"))


def generated_record(instruction="What does functional testing check?", expected="It checks feature workflows.", source_ref="source_1"):
    quote = source().elements[0].text
    return {"instruction": instruction, "context": quote, "expected_output": expected, "category": "testing", "difficulty": "easy", "source_refs": [source_ref], "evidence": [{"source_ref": source_ref, "quote": quote}]}


class ReviewProvider:
    def __init__(self, review_output, revision_output=None):
        self.calls = []
        self.review_output = review_output
        self.revision_output = revision_output

    def generate(self, *, messages, schema, max_tokens):
        self.calls.append(messages)
        if schema.get("properties", {}).get("issues"):
            return ProviderJob(f"review-{len(self.calls)}", "completed", self.review_output)
        if self.revision_output is not None and len(self.calls) > 1:
            return ProviderJob(f"revision-{len(self.calls)}", "completed", self.revision_output)
        return ProviderJob(f"generation-{len(self.calls)}", "completed", {"records": [generated_record()]})


def run_pipeline(tmp_path: Path, provider: ReviewProvider, *, review_output=None, revision_output=None):
    provider.review_output = review_output or {"issues": [], "summary": "No actionable quality issues were identified."}
    provider.revision_output = revision_output
    generator = RunPodDatasetGenerator(provider, max_model_len=4096, records_per_batch=2, max_dataset_records=2)
    reviewer = QualityReviewService(provider, model="gpt-oss-20b", max_model_len=4096)
    from app.services.quality_review import QualityRevisionService
    reviser = QualityRevisionService(provider, model="gpt-oss-20b")
    pipeline = PipelineService(tmp_path, generator=generator, quality_review_enabled=True, quality_reviewer=reviewer, quality_reviser=reviser)
    return pipeline.run(PipelineInput(job_id="gen", source_path=tmp_path / "source.txt", source_filename="source.txt", dataset_prompt="Create 1 record", output_format=OutputFormat.JSON), extracted_document=source(), file_id="file", model="gpt-oss-20b")


def test_no_revision_path_packages_and_has_no_revision_call(tmp_path):
    provider = ReviewProvider(None)
    result = run_pipeline(tmp_path, provider)
    assert result.quality_review.status == "passed"
    assert result.quality_review.revision_attempted is False
    assert len(provider.calls) == 2
    assert (tmp_path / "gen" / "dataset.zip").exists()


def test_warnings_allow_package_without_revision(tmp_path):
    provider = ReviewProvider(None)
    review = {"issues": [{"code": "REPETITIVE_RECORDS", "severity": "blocking", "record_ids": ["gen-1"], "message": "Examples repeat the same task pattern.", "suggested_action": "Vary the testing concept."}], "summary": "One issue."}
    result = run_pipeline(tmp_path, provider, review_output=review)
    assert result.quality_review.status == "passed_with_warnings"
    assert result.quality_review.blocking_issues == 0
    assert result.quality_review.revision_attempted is False
    assert len(provider.calls) == 2


def test_blocking_review_uses_one_targeted_revision_and_revalidates(tmp_path):
    provider = ReviewProvider(None)
    review = {"issues": [{"code": "SOURCE_SUPPORT_CONCERN", "severity": "warning", "record_ids": ["gen-1"], "message": "The answer needs tighter source support.", "suggested_action": "Use the supplied evidence directly."}], "summary": "Improve the answer."}
    revised = {"records": [generated_record(expected="It checks feature workflows and expected functionality.", source_ref="s1")]}
    result = run_pipeline(tmp_path, provider, review_output=review, revision_output=revised)
    assert result.quality_review.revision_attempted is True
    assert result.quality_review.revision_succeeded is True
    assert result.validation_report.grounding.grounded_records == 1
    assert len(provider.calls) == 3


def test_failed_revision_revalidation_blocks_package_and_never_retries(tmp_path):
    provider = ReviewProvider(None)
    review = {"issues": [{"code": "SOURCE_SUPPORT_CONCERN", "severity": "blocking", "record_ids": ["gen-1"], "message": "Evidence needs correction.", "suggested_action": "Use the exact source quote."}], "summary": "One blocking issue."}
    invalid_revision = {"records": [{**generated_record(source_ref="s1"), "evidence": [{"source_ref": "s1", "quote": "Not in source text."}]}]}
    with pytest.raises(ValueError):
        run_pipeline(tmp_path, provider, review_output=review, revision_output=invalid_revision)
    assert len(provider.calls) == 3


def test_review_rejects_unknown_ids_and_authority_fields():
    with pytest.raises(ProviderError):
        _parse_review_output({"issues": [{"code": "REPETITIVE_RECORDS", "severity": "warning", "record_ids": ["unknown"], "message": "x", "suggested_action": "y"}], "summary": "x"}, {"known"})
    with pytest.raises(ProviderError):
        _parse_review_output({"issues": [], "summary": "x", "passed": True}, set())


def test_review_issue_cap_and_message_bounds():
    issues = [{"code": "REPETITIVE_RECORDS", "severity": "warning", "record_ids": [], "message": "x", "suggested_action": "y"} for _ in range(26)]
    with pytest.raises(ProviderError):
        _parse_review_output({"issues": issues, "summary": "x"}, set())
