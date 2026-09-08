from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import valid_record
from dataset_forge_critic.artifacts import export_adapter, sync_artifacts
from dataset_forge_critic.checkpointing import compatibility_identity, validate_resume
from dataset_forge_critic.cli import _configuration
from dataset_forge_critic.integrity import derive_jsonl, safe_extract_tar, sha256_file
from dataset_forge_critic.metrics import evaluate_predictions
from dataset_forge_critic.mixture import MixedRecordStream
from dataset_forge_critic.modeling import discover_lora_modules
from dataset_forge_critic.runtime import RunLock, request_stop
from dataset_forge_critic.synthetic import run_synthetic
from dataset_forge_critic.tokenization import TokenizationExclusion, pad_batch, tokenize_record

ROOT = Path(__file__).resolve().parents[1]


class FakeProcessor:
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        @staticmethod
        def decode(values, skip_special_tokens=True):
            token = values[0]
            return {1: "<eos>", 106: "<turn|>", 107: "\n"}.get(token, chr(token - 1000) if token >= 1000 else "x")

    tokenizer = Tokenizer()

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict):
        assert tokenize and return_dict
        prompt = messages[0]["content"][0]["text"]
        prefix = [1] + [10 + sum(map(ord, prompt[index:index + 8])) % 900 for index in range(0, len(prompt), 8)] + [99]
        if len(messages) == 1:
            assert add_generation_prompt
            return {"input_ids": prefix}
        assert not add_generation_prompt
        completion = messages[-1]["content"][0]["text"]
        answer = [1000 + ord(char) for char in completion] + [106, 107]
        return {"input_ids": prefix + answer}

    def decode(self, values, skip_special_tokens=True):
        return ""


def decision_record():
    record = valid_record()
    record["target"].update({"decision": "REJECT", "issue_codes": ["GROUNDING_CONTRADICTION"], "critique": "The answer conflicts with the source."})
    record["supervision"]["available_targets"].extend(["decision", "issue_codes", "critique"])
    return record


def _write_rows(path: Path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8", newline="\n")


def test_completion_mask_and_padding_are_exact():
    tokenized = tokenize_record(decision_record(), "public", FakeProcessor(), 4096)
    assert tokenized.supervised_tokens > 0
    assert all(value == -100 for value in tokenized.labels[:tokenized.prompt_tokens])
    assert tokenized.labels[tokenized.prompt_tokens:] == tokenized.input_ids[tokenized.prompt_tokens:]
    assert tokenized.input_ids[-2:] == [106, 1]
    padded = pad_batch([tokenized], 0)
    assert padded["labels"][0] == tokenized.labels


def test_truncation_is_rejected_not_silent():
    with pytest.raises(TokenizationExclusion, match="SEQUENCE_TOO_LONG"):
        tokenize_record(decision_record(), "public", FakeProcessor(), 2)


def test_sampler_is_reproducible_and_resume_exact(tmp_path):
    public, native = tmp_path / "public.jsonl", tmp_path / "native.jsonl"
    _write_rows(public, [{"id": index} for index in range(5)]); _write_rows(native, [{"id": index} for index in range(3)])
    one = MixedRecordStream(public, native, .5, 9); prefix = [one.next() for _ in range(7)]; state = one.state_dict(); suffix = [one.next() for _ in range(8)]
    two = MixedRecordStream(public, native, .5, 9); assert [two.next() for _ in range(7)] == prefix
    two.load_state_dict(state); assert [two.next() for _ in range(8)] == suffix


def test_synthetic_interrupted_resume_matches_uninterrupted(tmp_path):
    public, native = tmp_path / "p.jsonl", tmp_path / "n.jsonl"
    _write_rows(public, [{"synthetic_target": 0.0}] * 4); _write_rows(native, [{"synthetic_target": 1.0}] * 4)
    full = run_synthetic(public, native, tmp_path / "full.json", steps=20)
    run_synthetic(public, native, tmp_path / "resume.json", steps=7)
    resumed = run_synthetic(public, native, tmp_path / "resume.json", steps=20, resume=True)
    assert resumed["trajectory"] == full["trajectory"]


def test_resume_rejects_changed_identity():
    expected = compatibility_identity("a", "m", "r", {"x": "1"})
    with pytest.raises(ValueError, match="CHECKPOINT_INCOMPATIBLE"):
        validate_resume({"compatibility": {**expected, "revision": "other"}}, expected)


def test_lora_discovery_only_matches_text_modules():
    torch = pytest.importorskip("torch")
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.language_model = torch.nn.Sequential(torch.nn.Linear(2, 2)); self.vision_tower = torch.nn.Linear(2, 2)
    assert discover_lora_modules(Tiny(), r"(^|\.)(language_model|text_model)(\.|$)") == ["language_model.0"]
    with pytest.raises(ValueError, match="LORA_TARGET_DISCOVERY_EMPTY"): discover_lora_modules(Tiny(), r"never")


def test_metrics_known_native_answers_and_missing_labels():
    target = valid_record(); target["task_family"] = "dataset_forge_native"; target["target"] = {"decision": "REJECT", "scores": {}, "issue_codes": []}; target["supervision"] = {"available_targets": ["decision"]}
    envelope = {"decision": "accept", "confidence": 1.0, "reason_codes": [], "feedback": None, "model_version": "dataset-forge-critic-v1"}
    result = evaluate_predictions([("native", target, json.dumps(envelope, separators=(",", ":")))])
    assert result["native"]["false_acceptance_of_reject"] == 1.0
    assert result["preference_accuracy"] is None and result["score_mae"] == {}


def test_physical_lf_export_serializes_schema_and_hash(tmp_path):
    source, destination = tmp_path / "source.jsonl", tmp_path / "derived.jsonl"
    record = valid_record(); record.pop("schema_version", None)
    source.write_bytes((json.dumps(record) + "\r\n").encode())
    manifest = derive_jsonl(source, destination, source_identity="fixture")
    assert b"\r\n" not in destination.read_bytes()
    assert json.loads(destination.read_text())["schema_version"] == "1.0.0"
    assert manifest["output_sha256"] == sha256_file(destination)


def test_safe_archive_rejects_traversal(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        info = tarfile.TarInfo("../escape"); payload = b"bad"; info.size = len(payload); output.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="UNSAFE_ARCHIVE_MEMBER"):
        safe_extract_tar(archive, tmp_path / "out", sha256_file(archive))


def test_run_lock_duplicate_and_graceful_stop(tmp_path):
    lock = RunLock(tmp_path, "digest"); lock.acquire()
    try:
        with pytest.raises(ValueError, match="RUN_ALREADY_ACTIVE"): RunLock(tmp_path, "digest").acquire()
        assert request_stop(tmp_path)["status"] == "GRACEFUL_STOP_REQUESTED"
        assert (tmp_path / "STOP_REQUESTED").is_file()
    finally: lock.release()


def test_profiles_are_strict_and_confirmation_digests_stable():
    digests = []
    for path in sorted((ROOT / "configs" / "profiles").glob("*.yaml")):
        config = _configuration(path); digests.append(config.digest()); assert config.packing is False
        assert config.model.revision == "ee0ef6023621cff504d758262d4e04895a5af4a2"
    assert len(digests) == 5 and len(set(digests)) == 5


def test_adapter_export_manifest_is_complete(tmp_path):
    config = _configuration(ROOT / "configs" / "profiles" / "gpu-smoke.yaml")
    checkpoint = tmp_path / "checkpoint"; (checkpoint / "adapter").mkdir(parents=True); (checkpoint / "adapter" / "adapter_config.json").write_text("{}")
    identity = compatibility_identity(config.digest(), config.model.model_id, config.model.revision, {"data": "hash"})
    (checkpoint / "checkpoint-manifest.json").write_text(json.dumps({"step": 3, "compatibility": identity}))
    result = export_adapter(checkpoint, tmp_path / "export", config)
    assert result["adapter_only"] is True and result["standalone_model"] is False and result["reload_verification"] == "GPU_PENDING"


def test_local_sync_verifies_and_remote_retry_failure_is_actionable(tmp_path, monkeypatch):
    source = tmp_path / "source"; source.mkdir(); (source / "x").write_text("x")
    assert sync_artifacts(source, str(tmp_path / "copy"), visibility=None, retries=0)["status"] == "LOCAL_SYNC_VERIFIED"
    calls = []
    monkeypatch.setattr("dataset_forge_critic.artifacts.subprocess.run", lambda *a, **k: calls.append(1) or SimpleNamespace(returncode=1, stderr="denied", stdout=""))
    monkeypatch.setattr("dataset_forge_critic.artifacts.time.sleep", lambda _: None)
    with pytest.raises(ValueError, match="REMOTE_SYNC_FAILED"): sync_artifacts(source, "hf://owner/repo", visibility="private", retries=2)
    assert len(calls) == 3


def test_notebook_cannot_start_from_run_all():
    notebook = json.loads((ROOT / "notebooks" / "critic-control.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    assert "dataset-forge-critic launch" not in sources[0]
    assert any("Start Training" in source and "confirmation" in source for source in sources)
