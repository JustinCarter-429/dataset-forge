import json
import httpx
import pytest

from app.providers.contracts import ProviderConfig, ProviderError
from app.providers.runpod import RunPodProvider
from app.services.context_projection import build_context_batches
from app.providers.contracts import ProviderJob
from app.services.generation import RunPodDatasetGenerator, requested_record_count
from app.services.job_store import InMemoryJobStore
from app.domain.models import GenerationJob
from app.utils.files import safe_download_name
from app.domain.extraction_models import CanonicalExtractedDocument, ExtractionElement, ExtractionElementType, ExtractionStatistics, ExtractionValidation
from app.domain.enums import OutputFormat
from app.domain.models import CanonicalDataset, PipelineInput
from app.services.pipeline import PipelineService
from app.services.validation import Phase4ValidationService


def multibatch_document():
    elements = [ExtractionElement(elementId=f"canonical-{i}", type=ExtractionElementType.PARAGRAPH, text=f"Evidence {i} is authoritative.", order=i) for i in range(8)]
    return CanonicalExtractedDocument(documentId="d", sourceFileId="f", sourceFilename="x.txt", mimeType="text/plain", extractor="plain_text", extractorVersion="1", blocks=elements, statistics=ExtractionStatistics(pageCount=None, characterCount=200, wordCount=40, elementCount=8, headingCount=0, paragraphCount=8, listItemCount=0, tableCount=0, tableRowCount=0, nonEmptyElementCount=8), validation=ExtractionValidation(valid=True, quality="good"))


def provider_record(ref, quote, temporary_id="record-1"):
    return {"instruction": "What is the evidence?", "context": quote, "expected_output": quote, "category": "testing", "difficulty": "easy", "source_refs": [ref], "evidence": [{"source_ref": ref, "quote": quote}], "metadata": {"record_id": temporary_id}}


def test_status_transient_error_retries_same_external_job_without_resubmitting():
    calls = []
    statuses = iter([httpx.Response(503), httpx.Response(200, json={"status": "IN_QUEUE"}), httpx.Response(200, json={"status": "COMPLETED", "output": {"records": []}})])
    def handler(request):
        calls.append(request.url.path)
        if request.method == "POST": return httpx.Response(200, json={"id": "same-job"})
        return next(statuses)
    config = ProviderConfig("endpoint", "secret", "model", poll_interval_seconds=.001, queue_timeout_seconds=1, execution_timeout_seconds=1)
    provider = RunPodProvider(config, httpx.Client(transport=httpx.MockTransport(handler)))
    provider.generate(messages=[{"role": "user", "content": "x"}], schema={"type": "object"}, max_tokens=32)
    assert calls.count("/v2/endpoint/run") == 1
    assert calls.count("/v2/endpoint/status/same-job") == 3
    assert provider.metrics["providerJobsCreated"] == 1
    assert provider.metrics["providerStatusPolls"] == 3
    assert provider.metrics["providerTransportRetries"] == 1


def test_batch_planner_allocates_every_nonempty_source_unit():
    from app.domain.extraction_models import CanonicalExtractedDocument, ExtractionElement, ExtractionElementType, ExtractionStatistics, ExtractionValidation
    elements = [ExtractionElement(elementId=f"s{i}", type=ExtractionElementType.PARAGRAPH, text="x" * 500, order=i) for i in range(12)]
    doc = CanonicalExtractedDocument(documentId="d", sourceFileId="f", sourceFilename="x.txt", mimeType="text/plain", extractor="plain_text", extractorVersion="1", blocks=elements, statistics=ExtractionStatistics(pageCount=None, characterCount=6000, wordCount=600, elementCount=12, headingCount=0, paragraphCount=12, listItemCount=0, tableCount=0, tableRowCount=0, nonEmptyElementCount=12), validation=ExtractionValidation(valid=True, quality="good"))
    batches = build_context_batches(doc, max_model_len=1024, records_per_batch=2, record_limit=12)
    assert len(batches) > 1
    assert [item for batch in batches for item in batch.source_element_ids] == [f"s{i}" for i in range(12)]
    assert all(batch.estimated_input_tokens <= 1024 for batch in batches)


def test_descriptive_record_count_still_produces_multibatch_plan():
    assert requested_record_count("Create 8 source-grounded software-testing records.", 20) == 8
    document = multibatch_document()
    extra = [ExtractionElement(elementId=f"canonical-{i}", type=ExtractionElementType.PARAGRAPH, text=f"Evidence {i} is authoritative.", order=i) for i in range(12)]
    document = document.model_copy(update={"elements": extra})
    assert len(build_context_batches(document, max_model_len=2048, records_per_batch=4, record_limit=8)) == 2


def test_cancel_is_idempotent_and_never_marks_package_ready():
    store = InMemoryJobStore()
    store.add_job(GenerationJob(id="a" * 32, status="generating", stage="generating", progress={"percent": 40, "currentStage": "generating"}, file={"id": "f", "name": "source.txt"}, output={"requestedFormat": "json", "recordCount": None, "finalRecordCount": None, "sizeBytes": None}, capabilities={}))
    first = store.request_cancel("a" * 32)
    second = store.request_cancel("a" * 32)
    assert first.status == second.status == "cancelled"
    assert first.package_ready is False


def test_download_name_is_safe_and_meaningful():
    assert safe_download_name(r"..\private\qa source?.txt") == "qa-source-dataset.zip"


def test_batch_aliases_are_immutable_and_canonicalized_before_assembly():
    doc = multibatch_document()
    batches = build_context_batches(doc, max_model_len=2048, records_per_batch=2, record_limit=4)
    assert len(batches) == 2
    assert batches[0].alias_to_canonical["source_1"] != batches[1].alias_to_canonical["source_1"]

    class AliasProvider:
        def __init__(self): self.calls = 0
        def generate(self, *, messages, schema, max_tokens):
            self.calls += 1
            quote = "Evidence 0 is authoritative." if self.calls == 1 else "Evidence 4 is authoritative."
            return ProviderJob(f"job-{self.calls}", "completed", {"records": [{"instruction": "What is the evidence?", "context": quote, "expected_output": quote, "category": "testing", "difficulty": "easy", "source_refs": ["source_1"], "evidence": [{"source_ref": "source_1", "quote": quote}]}]})
    dataset = RunPodDatasetGenerator(AliasProvider(), max_model_len=2048, records_per_batch=2, max_dataset_records=4).generate(doc, "Create 4 records", generation_id="g", file_id="f", model="m")
    assert [record.source_refs for record in dataset.records] == [["canonical-0"], ["canonical-4"]]
    assert [record.metadata["record_id"] for record in dataset.records] == ["g-1", "g-2"]


def test_cross_batch_canonical_reference_fails_closed():
    doc = multibatch_document()
    class BadProvider:
        def __init__(self): self.calls = 0
        def generate(self, *, messages, schema, max_tokens):
            self.calls += 1
            ref = "canonical-0" if self.calls >= 2 else "source_1"
            quote = "Evidence 0 is authoritative."
            return ProviderJob(f"job-{self.calls}", "completed", {"records": [{"instruction": "i", "context": quote, "expected_output": "o", "category": "testing", "difficulty": "easy", "source_refs": [ref], "evidence": [{"source_ref": ref, "quote": quote}]}]})
    with pytest.raises(Exception, match="UNKNOWN_OR_CROSS_BATCH_SOURCE_REF"):
        RunPodDatasetGenerator(BadProvider(), max_model_len=2048, records_per_batch=2, max_dataset_records=4).generate(doc, "Create 4 records", generation_id="g", file_id="f", model="m")


def test_actual_alias_reuse_failure_fixture_fails_before_fix_and_passes_after_fix():
    """The retained Phase 6 failure class: two batches both returned source_1."""
    doc = multibatch_document()
    # This is the historical global assembly shape, before a batch-local alias
    # was resolved. Phase 4 correctly rejects both records as unknown sources.
    from app.services.generation import _normalize_provider_records
    raw = [provider_record("source_1", "Evidence 0 is authoritative."), provider_record("source_1", "Evidence 4 is authoritative.")]
    legacy = _normalize_provider_records(raw, "g", "f", "m")
    _, legacy_report = Phase4ValidationService().validate(CanonicalDataset(records=legacy), doc)
    assert legacy_report.status == "failed"
    assert {issue.code for issue in legacy_report.issues} >= {"INVALID_SOURCE_REF", "EVIDENCE_NOT_FOUND"}

    class ReusedAliasProvider:
        def __init__(self): self.calls = 0
        def generate(self, *, messages, schema, max_tokens):
            self.calls += 1
            return ProviderJob(f"job-{self.calls}", "completed", {"records": [provider_record("source_1", f"Evidence {0 if self.calls == 1 else 4} is authoritative.")]})
    final = RunPodDatasetGenerator(ReusedAliasProvider(), max_model_len=2048, records_per_batch=2, max_dataset_records=4).generate(doc, "Create 4 records", generation_id="g", file_id="f", model="m")
    _, report = Phase4ValidationService().validate(final, doc)
    assert report.grounding.status == "passed"


@pytest.mark.parametrize("ref", ["source_99", "canonical-0"])
def test_unknown_or_canonical_provider_alias_is_rejected(ref):
    class BadProvider:
        def generate(self, *, messages, schema, max_tokens):
            return ProviderJob("job", "completed", {"records": [provider_record(ref, "Evidence 0 is authoritative.")]})
    with pytest.raises(Exception, match="UNKNOWN_OR_CROSS_BATCH_SOURCE_REF"):
        RunPodDatasetGenerator(BadProvider(), max_model_len=2048, records_per_batch=2, max_dataset_records=4).generate(multibatch_document(), "Create 4 records", generation_id="g", file_id="f", model="m")


def test_evidence_paired_with_wrong_source_and_ambiguous_refs_fail_closed():
    generator = RunPodDatasetGenerator(object(), max_model_len=2048, records_per_batch=2, max_dataset_records=4)
    batch = build_context_batches(multibatch_document(), max_model_len=2048, records_per_batch=2, record_limit=4)[0]
    wrong = provider_record("source_1", "Evidence 1 is authoritative.")
    with pytest.raises(ValueError, match="BATCH_EVIDENCE_NOT_FOUND"):
        generator._canonicalize_source_refs([wrong], batch)
    ambiguous = provider_record("source_1", "Evidence 0 is authoritative.")
    ambiguous["source_refs"] = ["source_1", "source_2"]
    with pytest.raises(ValueError, match="SOURCE_REF_EVIDENCE_MISMATCH"):
        generator._canonicalize_source_refs([ambiguous], batch)


def test_duplicate_provider_temporary_ids_are_rebased_to_global_unique_ids():
    doc = multibatch_document()
    class DuplicateTemporaryIdProvider:
        def __init__(self): self.calls = 0
        def generate(self, *, messages, schema, max_tokens):
            self.calls += 1
            index = 0 if self.calls == 1 else 4
            return ProviderJob(f"job-{self.calls}", "completed", {"records": [provider_record("source_1", f"Evidence {index} is authoritative.", "provider-record-1")]})
    final = RunPodDatasetGenerator(DuplicateTemporaryIdProvider(), max_model_len=2048, records_per_batch=2, max_dataset_records=4).generate(doc, "Create 4 records", generation_id="g", file_id="f", model="m")
    assert [record.metadata["record_id"] for record in final.records] == ["g-1", "g-2"]
    assert len({record.metadata["record_id"] for record in final.records}) == len(final.records)


def test_multibatch_partial_grounding_failure_never_starts_quality_review_or_package(tmp_path):
    doc = multibatch_document()
    class FailingSecondBatchProvider:
        def __init__(self): self.calls = 0
        def generate(self, *, messages, schema, max_tokens):
            self.calls += 1
            if self.calls == 1:
                return ProviderJob("generation-1", "completed", {"records": [provider_record("source_1", "Evidence 0 is authoritative.")]})
            return ProviderJob(f"generation-{self.calls}", "completed", {"records": [provider_record("source_1", "Evidence 0 is authoritative.")]})
    class NeverReview:
        calls = 0
        def review(self, *args):
            self.calls += 1
            raise AssertionError("quality review must not execute before global grounding")
    provider = FailingSecondBatchProvider()
    generator = RunPodDatasetGenerator(provider, max_model_len=2048, records_per_batch=2, max_dataset_records=4)
    reviewer = NeverReview()
    pipeline = PipelineService(tmp_path, generator=generator, quality_review_enabled=True, quality_reviewer=reviewer, quality_reviser=object())
    with pytest.raises(ProviderError):
        pipeline.run(PipelineInput(job_id="g", source_path=tmp_path / "source.txt", source_filename="source.txt", dataset_prompt="Create 4 records", output_format=OutputFormat.JSON), extracted_document=doc, file_id="f", model="m")
    assert reviewer.calls == 0
    assert not (tmp_path / "g" / "dataset.zip").exists()
