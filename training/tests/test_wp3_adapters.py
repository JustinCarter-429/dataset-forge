from __future__ import annotations

from dataset_forge_critic.adapters import FactsGroundingAdapter, HaluBenchAdapter, HelpSteer2Adapter, UltraFeedbackAdapter
from dataset_forge_critic.schemas import TaskFamily


def test_facts_is_input_only_grounding():
    record = FactsGroundingAdapter("snap").transform_validated({"_row": 1, "system_instruction": "sys", "user_request": "question", "context_document": "context", "full_prompt": "full"})[0]
    assert record.task_family is TaskFamily.GROUNDING
    assert record.supervision.available_targets == []


def test_halubench_preserves_verified_polarity():
    record = HaluBenchAdapter("snap").transform_validated({"_row": 1, "id": "x", "passage": "p", "question": "q", "answer": "a", "label": "PASS", "source_ds": "s", "score": 1})[0]
    assert record.target.scores.hallucination.value == 1
    assert record.task_family is TaskFamily.HALLUCINATION_DETECTION


def test_helpsteer2_preference_polarity_and_zero_strength():
    adapter = HelpSteer2Adapter("snap", "preference")
    base = {"_row": 4, "split": "train", "prompt": "p", "response_1": "one", "response_2": "two", "preference_elaboration": "reason"}
    assert adapter.transform_validated(base | {"preference_strength": -2})[0].target.preferred_candidate_index == 0
    assert adapter.transform_validated(base | {"preference_strength": 3})[0].target.preferred_candidate_index == 1
    assert adapter.transform_validated(base | {"preference_strength": 0})[0].target.preferred_candidate_index is None


def test_helpsteer2_quality_preserves_source_scale():
    adapter = HelpSteer2Adapter("snap", "train")
    record = adapter.transform_validated({"_row": 2, "prompt": "p", "response": "r", "helpfulness": 0, "correctness": 1, "coherence": 2, "complexity": 3, "verbosity": 4})[0]
    assert record.target.scores.verbosity.value == 4


def test_ultrafeedback_emits_one_record_per_completion():
    adapter = UltraFeedbackAdapter("snap", "flan")
    records = adapter.transform_validated({"_row": 9, "instruction": "do", "completions": [{"response": "first", "overall_score": 3, "critique": "ok"}, {"response": "second"}]})
    assert len(records) == 2
    assert records[0].target.scores.usefulness.value == 3
    assert records[0].record_id != records[1].record_id
