from pathlib import Path

import json

from rubric_dpo.cli.launch_matrix import Task, _finalize_pilot, _latest_checkpoint


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
