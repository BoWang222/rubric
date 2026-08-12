import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rubric_dpo.checkpointing import seal_local_checkpoint
from rubric_dpo.cli.launch_matrix import (
    Task,
    _finalize_pilot,
    _latest_checkpoint,
    _parse_seeds,
    _parse_variants,
    _train_command,
)


def test_parse_single_and_multiple_full_run_seeds() -> None:
    assert _parse_seeds("42") == (42,)
    assert _parse_seeds("13,42,100") == (13, 42, 100)


@pytest.mark.parametrize("value", ["", "42,42", "-1", "seed42"])
def test_reject_invalid_full_run_seeds(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_seeds(value)


def test_parse_selected_full_run_variants() -> None:
    assert _parse_variants("dpo") == ("dpo",)
    assert _parse_variants("mmpo,odpo_loggap") == ("mmpo", "odpo_loggap")


@pytest.mark.parametrize("value", ["", "dpo,dpo", "not-a-baseline"])
def test_reject_invalid_full_run_variants(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_variants(value)


def test_latest_checkpoint_is_bounded_by_declared_final_step(tmp_path: Path) -> None:
    output = tmp_path / "run"
    for step in (8, 283, 999):
        checkpoint = output / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
        seal_local_checkpoint(checkpoint, output)
    (output / "checkpoint-not-a-step").mkdir()
    task = Task("dpo", output, 1e-6, max_steps=566)
    assert _latest_checkpoint(task) == output / "checkpoint-283"


def test_latest_checkpoint_ignores_unsealed_partial_save(tmp_path: Path) -> None:
    output = tmp_path / "run"
    checkpoint = output / "checkpoint-283"
    (checkpoint / "pytorch_model_fsdp_0").mkdir(parents=True)
    task = Task("dpo", output, 1e-6, max_steps=566)
    assert _latest_checkpoint(task) is None


def test_latest_checkpoint_is_none_for_fresh_run(tmp_path: Path) -> None:
    task = Task("dpo", tmp_path / "fresh", 1e-6, max_steps=16)
    assert _latest_checkpoint(task) is None


def test_eight_gpu_command_preserves_effective_batch_64(tmp_path: Path) -> None:
    args = SimpleNamespace(
        root=tmp_path,
        model=tmp_path / "model",
        dataset_dir=tmp_path / "dataset",
        reference_cache=tmp_path / "cache",
        two_gpu_offload=False,
    )
    task = Task("dpo", tmp_path / "run", 1e-6)
    command = _train_command(args, task, gpu_count=8, port=29500)
    assert str(tmp_path / "configs/accelerate/fsdp_8gpu.yaml") in command
    per_device = command.index("--per-device-batch-size")
    accumulation = command.index("--gradient-accumulation-steps")
    assert command[per_device + 1] == "2"
    assert command[accumulation + 1] == "4"


def test_metrics_only_pilot_disables_checkpoint_saves(tmp_path: Path) -> None:
    args = SimpleNamespace(
        root=tmp_path,
        model=tmp_path / "model",
        dataset_dir=tmp_path / "dataset",
        reference_cache=tmp_path / "cache",
        two_gpu_offload=False,
    )
    task = Task("dpo", tmp_path / "pilot", 5e-7, max_steps=128, save_checkpoints=False, evaluate=True)
    command = _train_command(args, task, gpu_count=4, port=29700)
    assert "--no-save" in command
    assert "--save-steps" not in command
    assert "--evaluate-after-train" in command


def test_two_gpu_no_offload_command_preserves_effective_batch_64(tmp_path: Path) -> None:
    args = SimpleNamespace(
        root=tmp_path,
        model=tmp_path / "model",
        dataset_dir=tmp_path / "dataset",
        reference_cache=tmp_path / "cache",
        two_gpu_offload=False,
    )
    task = Task("dpo", tmp_path / "pilot", 5e-7, max_steps=64, save_checkpoints=False, evaluate=True)
    command = _train_command(args, task, gpu_count=2, port=29700)
    assert str(tmp_path / "configs/accelerate/fsdp_2gpu.yaml") in command
    per_device = command.index("--per-device-batch-size")
    accumulation = command.index("--gradient-accumulation-steps")
    assert command[per_device + 1] == "1"
    assert command[accumulation + 1] == "32"


def test_pilot_finalization_keeps_metrics_and_removes_recovery(tmp_path: Path) -> None:
    output = tmp_path / "pilot"
    checkpoint = output / "checkpoint-128"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trusted_local_checkpoint.json").write_text("{}")
    (output / "final_merged").mkdir()
    task = Task("dpo", output, 5e-7, max_steps=128, save_steps=128, evaluate=True)
    (output / "run_manifest.json").write_text(json.dumps({"status": "complete", "global_step": 128}))
    (output / "resolved_config.json").write_text(json.dumps({"variant": "dpo"}))
    (output / "metrics.json").write_text(json.dumps({
        "global_step": 128,
        "eval_baseline_loss_unweighted_dpo": 0.6,
        "eval_rewards/accuracies": 0.75,
    }))
    _finalize_pilot(task)
    report = json.loads((output / "pilot_verification.json").read_text())
    assert report["status"] == "verified_metrics_only"
    assert report["recovery_deleted"] is True
    assert report["validation_nll"] == 0.6
    assert not checkpoint.exists()
    assert not (output / "final_merged").exists()
