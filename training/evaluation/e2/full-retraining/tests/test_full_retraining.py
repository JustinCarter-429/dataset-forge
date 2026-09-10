from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TRAIN = load("e2_train", ROOT / "e2_train.py")
PREFLIGHT = load("e2_preflight", ROOT / "preflight.py")
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def test_configuration_freezes_full_identity_and_all_hyperparameters():
    assert CONFIG["schema_version"] == "phase-e2.4-full-retraining-v1"
    assert CONFIG["model"]["id"] == "google/gemma-4-E4B-it"
    assert CONFIG["model"]["revision"] == "ee0ef6023621cff504d758262d4e04895a5af4a2"
    assert CONFIG["data"]["train_records"] == 6350 and CONFIG["data"]["validation_records"] == 806
    opt = CONFIG["optimization"]
    expected = {"epochs": 3, "micro_batch_size": 1, "gradient_accumulation_steps": 8, "effective_batch_size": 8,
                "max_optimizer_steps": 2382, "learning_rate": 0.0001, "weight_decay": 0.0, "optimizer": "paged_adamw_8bit",
                "scheduler": "cosine", "warmup_ratio": 0.03, "max_grad_norm": 1.0, "precision": "bfloat16",
                "quantization": "4bit-nf4-double-quant", "lora_rank": 16, "lora_alpha": 16, "lora_dropout": 0.05,
                "max_sequence_length": 2048, "seed": 42}
    for key, value in expected.items():
        assert opt[key] == value


def test_exact_step_count_and_partial_final_batch():
    assert TRAIN.optimizer_step_count(6350, 3, 8) == 2382
    report = TRAIN.sampler_dry_run([f"r{i}" for i in range(6350)], 3, 8, 42)
    assert report["examples"] == 19050
    assert report["full_effective_batches"] == 2381
    assert report["final_partial_batch_examples"] == 2
    assert TRAIN.accumulation_target(19050, 19048, 8) == 2
    assert TRAIN.accumulation_target(19050, 19049, 8) == 2


def test_three_epoch_sampler_is_deterministic_without_replacement():
    ids = [f"r{i}" for i in range(101)]
    first = TRAIN.sampler_dry_run(ids, 3, 8, 42)
    second = TRAIN.sampler_dry_run(ids, 3, 8, 42)
    assert first == second and first["each_record_once_per_epoch"]
    assert len(set(first["epoch_order_sha256"])) == 3


def test_sampler_rejects_duplicate_record_ids():
    with pytest.raises(ValueError, match="DUPLICATE_CORPUS_RECORD_ID"):
        TRAIN.sampler_dry_run(["same", "same"], 3, 8, 42)


@pytest.mark.parametrize("resume_at", [6350, 6350 + 317])
def test_resume_at_epoch_boundary_and_mid_epoch_has_no_skip_or_repeat(resume_at: int):
    count, seed = 6350, 42
    complete = [TRAIN.sampler_coordinate(count, seed, index)[2] for index in range(count * 3)]
    resumed = ([TRAIN.sampler_coordinate(count, seed, index)[2] for index in range(resume_at)] +
               [TRAIN.sampler_coordinate(count, seed, index)[2] for index in range(resume_at, count * 3)])
    assert resumed == complete
    for epoch in range(3):
        segment = complete[epoch * count:(epoch + 1) * count]
        assert len(segment) == len(set(segment)) == count


def test_validation_and_checkpoint_schedules_include_final():
    assert TRAIN.schedule_steps(2382, 250, baseline=True) == [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2382]
    assert TRAIN.schedule_steps(2382, 250, baseline=False) == [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2382]


def test_best_checkpoint_selection_is_frozen_before_training():
    assert CONFIG["checkpoint_policy"]["best_selection"] == {"metric": "validation_loss", "direction": "minimize", "tie_breaker": "earliest_step"}
    assert CONFIG["checkpoint_policy"]["preserve_final_separately"] is True


def make_checkpoint(path: Path, digest: str = "digest") -> None:
    (path / "adapter").mkdir(parents=True)
    (path / "adapter/model.safetensors").write_bytes(b"adapter")
    files = TRAIN.checkpoint_file_hashes(path)
    TRAIN.atomic_json(path / "checkpoint-manifest.json", {"complete": True, "step": 8, "sample_index": 64,
                      "config_digest": digest, "metrics": {"validation_loss": 1.0}, "files": files})


def test_checkpoint_hash_verification_and_corruption_rejection(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    make_checkpoint(checkpoint)
    assert TRAIN.verify_checkpoint(checkpoint, "digest")["complete"]
    (checkpoint / "adapter/model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="CORRUPTED_CHECKPOINT"):
        TRAIN.verify_checkpoint(checkpoint, "digest")


def test_atomic_json_never_exposes_partial_document(tmp_path: Path):
    output = tmp_path / "progress.json"
    TRAIN.atomic_json(output, {"status": "running", "step": 1})
    assert json.loads(output.read_text()) == {"status": "running", "step": 1}
    assert not output.with_name("progress.json.tmp").exists()


def test_strict_parser_and_reason_code_metric_preserve_failures():
    valid = '{"decision":"accept","confidence":1.0,"reason_codes":[],"feedback":null,"model_version":"dataset-forge-critic-v1"}'
    assert TRAIN.parse_strict_response(valid)["decision"] == "accept"
    with pytest.raises(ValueError):
        TRAIN.parse_strict_response("```json\n" + valid + "\n```")
    assert TRAIN.reason_code_macro_f1([{"PROMPT_INJECTION"}], [set()]) < 1.0


def test_post_training_smoke_precedes_unchanged_held_out_thresholds():
    post = json.loads((ROOT / "post-training.json").read_text())
    assert post["sequence"].index("strict_contract_smoke_independent_cases") < post["sequence"].index("unchanged_105_case_e1_held_out_certification_if_smoke_passes")
    thresholds = ROOT.parents[1] / "thresholds.json"
    assert hashlib.sha256(thresholds.read_bytes()).hexdigest() == "74c6668fb05d43477952604d6a045b0f5d3b584332b87002aacef56d4271f98e"


def test_wrapper_is_authorization_duplicate_exit_code_and_sync_safe():
    text = (ROOT / "remote-runner.sh").read_text(encoding="utf-8")
    for required in ("E2_FULL_TRAINING_AUTHORIZED", "DUPLICATE_LAUNCH", "exit-code.tmp", "E2_SYNC_COMMAND", "sync_loop", "PACKAGE_HASH_MISMATCH"):
        assert required in text
    assert "--mode smoke" not in text and "--mode full" in text


def test_publication_scan_rejects_private_material():
    assert PREFLIGHT.publication_scan({"safe.txt": b"safe"})["status"] == "PASS"
    simulated_header = b"-----BEGIN " + b"PRIVATE KEY-----"
    assert PREFLIGHT.publication_scan({"bad.txt": simulated_header})["status"] == "FAIL"


def test_archive_safety_rejects_traversal(tmp_path: Path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        bundle.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="UNSAFE_ARCHIVE_MEMBER"):
        PREFLIGHT.verify_package(archive)


def test_retrieval_marker_requires_matching_hashes(tmp_path: Path):
    ledger = tmp_path / "ledger.json"
    marker = tmp_path / "RETRIEVAL_VERIFIED.json"
    ledger.write_text(json.dumps({"items": [{"remote_sha256": "a", "local_sha256": "a", "bytes": 1}]}))
    result = PREFLIGHT.write_retrieval_verified(ledger, marker)
    assert result["status"] == "VERIFIED" and result["instance_destruction_permitted"] is True
    ledger.write_text(json.dumps({"items": [{"remote_sha256": "a", "local_sha256": "b", "bytes": 1}]}))
    with pytest.raises(ValueError, match="RETRIEVAL_NOT_VERIFIED"):
        PREFLIGHT.write_retrieval_verified(ledger, marker)
