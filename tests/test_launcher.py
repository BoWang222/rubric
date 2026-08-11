from pathlib import Path
from types import SimpleNamespace

import json

from rubric_dpo.cli.launch_matrix import Task, _finalize_pilot, _latest_checkpoint, _run_process, _train_command


def test_latest_checkpoint_is_bounded_by_declared_final_step(tmp_path: Path) -> None:
    output = tmp_path / "run"
    for step in (8, 283, 999):
        (output / f"checkpoint-{step}").mkdir(parents=True)
    (output / "checkpoint-not-a-step").mkdir()
    task = Task("dpo", output, 1e-6, max_steps=566)
    assert _latest_checkpoint(task) == output / "checkpoint-283"


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


def test_eight_gpu_process_disables_nvls_by_default(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_run(command, *, env, stdout, stderr, check):
        captured.update(env)

    monkeypatch.delenv("NCCL_NVLS_ENABLE", raising=False)
    monkeypatch.setattr("rubric_dpo.cli.launch_matrix.subprocess.run", fake_run)
    log = tmp_path / "launcher.log"
    _run_process(["train"], "0,1,2,3,4,5,6,7", log)
    assert captured["NCCL_NVLS_ENABLE"] == "0"
    assert "NCCL_NVLS_ENABLE=0" in log.read_text()


def test_four_gpu_process_does_not_change_nvls(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_run(command, *, env, stdout, stderr, check):
        captured.update(env)

    monkeypatch.delenv("NCCL_NVLS_ENABLE", raising=False)
    monkeypatch.setattr("rubric_dpo.cli.launch_matrix.subprocess.run", fake_run)
    _run_process(["train"], "0,1,2,3", tmp_path / "launcher.log")
    assert "NCCL_NVLS_ENABLE" not in captured


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
