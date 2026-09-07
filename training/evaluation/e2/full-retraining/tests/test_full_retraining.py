from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("e2_train", ROOT / "e2_train.py")
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


def test_configuration_freezes_authorized_identity_and_hashes():
    value = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    assert value["model"] == {"id": "google/gemma-4-E4B-it", "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2", "architecture": "Gemma4ForConditionalGeneration", "renderer": "gemma4-critic-render-v2", "model_version": "dataset-forge-critic-v1"}
    assert value["data"]["package_sha256"] == "0ae841070c050a7ebe8f69fc85ad5d4de3aec146b861422ec72305489d44fe4b"
    assert value["data"]["train_sha256"] == "a7485d598825573e557a982dd4e86f7ba5868a9ee8d71c0306616b625b524ce8"
    assert value["data"]["validation_sha256"] == "4b991d36e2d179161293e96b81dbda1abff948fbb27ee7ad9e2634744cc31e47"


def test_effective_optimizer_budget_is_three_no_replacement_epochs():
    value = json.loads((ROOT / "config.json").read_text(encoding="utf-8")); opt = value["optimization"]
    assert opt["max_optimizer_steps"] * opt["effective_batch_size"] == value["data"]["train_records"] * opt["epochs"]
    assert opt["max_optimizer_steps"] <= opt["approved_step_ceiling"]


def test_stopping_tokens_are_pinned():
    value = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))["monitoring"]
    assert (value["eos_token_id"], value["assistant_turn_terminator_id"]) == (1, 106)


def test_stratified_subset_is_deterministic_and_balanced():
    items=[]
    for decision in ("accept", "revise", "reject"):
        for dataset_type in ("question_answer", "instruction_response", "classification", "scenario_expected_result", "custom"):
            for index in range(3):
                items.append({"canonical_record":{"record_id":f"{decision}-{dataset_type}-{index}","target":{"decision":decision},"input":{"dataset_spec":{"dataset_type":dataset_type}}}})
    first=MODULE.stratified(items,30,42); second=MODULE.stratified(items,30,42)
    assert first == second and len({(x['canonical_record']['target']['decision'],x['canonical_record']['input']['dataset_spec']['dataset_type']) for x in first}) == 15


def test_macro_f1_known_perfect_and_failed_values():
    truth=["accept","revise","reject"]
    assert MODULE.macro_f1(truth,truth) == 1.0
    assert MODULE.macro_f1(truth,[None,None,None]) == 0.0


def test_remote_wrapper_runs_smoke_before_full_and_is_duplicate_safe():
    text=(ROOT/'remote-runner.sh').read_text(encoding='utf-8')
    assert text.index('--mode smoke') < text.index('--mode full')
    assert 'DUPLICATE_LAUNCH' in text and 'exit-code.tmp' in text and 'STOP_REQUESTED' in text


def test_checkpoint_selection_uses_contract_and_safety_metrics():
    text=(ROOT/'e2_train.py').read_text(encoding='utf-8')
    for metric in ('strict_json_validity','accuracy','macro_f1','grounding_false_accept_rate','malicious_injection_failure_rate','parse_failure_rate','validation_loss'):
        assert metric in text
