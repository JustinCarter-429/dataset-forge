PROMPT_VERSION = "dataset-quality-review-v1"

SYSTEM_PROMPT = """You are Dataset Forge's bounded dataset quality reviewer.
Review only the supplied validated dataset records and their attached source evidence.
Return JSON matching the supplied schema. Do not regenerate records. Do not claim schema,
grounding, evidence, security, package, or pass/fail authority. Do not include reasoning,
confidence scores, praise, or fields outside the schema. Report only concrete actionable
quality issues and refer only to supplied record IDs. An empty issue list is valid."""


def user_prompt(dataset_prompt: str, review_input: str, batch_id: str, requested_records: int) -> str:
    return f"""Review batch {batch_id}. Review at most {requested_records} records.

Dataset request:
{dataset_prompt}

Bounded validated dataset and deterministic summaries:
{review_input}

Look for repetitive or redundant examples, low instruction diversity, weak context or
expected outputs, ambiguity, category/difficulty inconsistency, overly broad or trivial
answers, poor coverage, and unsupported-looking claims. Use only these issue codes:
REPETITIVE_RECORDS, LOW_INSTRUCTION_DIVERSITY, WEAK_CONTEXT, WEAK_EXPECTED_OUTPUT,
AMBIGUOUS_INSTRUCTION, DIFFICULTY_MISMATCH, CATEGORY_INCONSISTENCY, REDUNDANT_TEST_CASE,
SOURCE_SUPPORT_CONCERN, OVERLY_BROAD_ANSWER, OVERLY_TRIVIAL_RECORD, POOR_DATASET_COVERAGE,
OTHER_BOUNDED_QUALITY_ISSUE.

Severity is a proposal only; the application applies the final policy. Keep each message
and suggested action concise. Return only structured JSON."""


def revision_prompt(dataset_prompt: str, records: str, issues: str, source_context: str) -> str:
    return f"""Revise only the affected records in this dataset request:
{dataset_prompt}

Affected records:
{records}

Application-provided quality issues:
{issues}

Allowed source units and evidence context:
{source_context}

Return only the revised records in the canonical dataset schema. Preserve the supplied
record IDs through application lineage, use only allowed source_refs and evidence quotes,
do not create unsupported facts, do not revise unaffected records, do not self-grade, and
do not include reasoning or extra fields."""
