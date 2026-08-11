from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rubric_dpo.constants import VARIANTS
from rubric_dpo.utils import atomic_json, sha256_file


@dataclass(frozen=True)
class Task:
    variant: str
    output: Path
    lr: float
    gamma: float = 2.2
    alpha: float = 0.5
    seed: int = 13
    max_steps: int = 16
    save_steps: int = 8
    evaluate: bool = False
    eval_split: str = "validation"


def _train_command(args, task: Task, gpu_count: int, port: int, stop_after: int | None = None, resume: Path | None = None) -> list[str]:
    config_name = "fsdp_2gpu_cpu_offload.yaml" if gpu_count == 2 and args.two_gpu_offload else f"fsdp_{gpu_count}gpu.yaml"
    fsdp = args.root / "configs/accelerate" / config_name
    per_device, accumulation = (2, 8) if gpu_count == 4 else (1, 32)
    command = [
        str(Path(sys.executable).with_name("accelerate")), "launch", "--config_file", str(fsdp), "--main_process_port", str(port),
        "-m", "rubric_dpo.cli.train",
        "--variant", task.variant,
        "--model", str(args.model),
        "--dataset-dir", str(args.dataset_dir),
        "--reference-cache", str(args.reference_cache),
        "--output-dir", str(task.output),
        "--train-split", "smoke_1024" if task.max_steps == 16 else "train",
        "--seed", str(task.seed),
        "--learning-rate", str(task.lr),
        "--gamma", str(task.gamma),
        "--alpha", str(task.alpha),
        "--max-steps", str(task.max_steps),
        "--save-steps", str(task.save_steps),
        "--per-device-batch-size", str(per_device),
        "--gradient-accumulation-steps", str(accumulation),
        "--logging-steps", "1" if task.max_steps <= 16 else "10",
    ]
    if task.evaluate:
        command += ["--evaluate-after-train", "--eval-split", task.eval_split]
    if stop_after is not None:
        command += ["--stop-after-step", str(stop_after)]
    if resume is not None:
        command += ["--resume-from-checkpoint", str(resume)]
    return command


def _completed(task: Task) -> bool:
    path = task.output / "run_manifest.json"
    if not path.exists():
        return False
    manifest = json.loads(path.read_text())
    return manifest.get("status") == "complete" and int(manifest.get("global_step", -1)) == task.max_steps


def _latest_checkpoint(task: Task) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in task.output.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and path.is_dir() and int(match.group(1)) <= task.max_steps:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _run_process(command: list[str], gpus: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def _gpu_ids(args, count: int | None = None) -> list[str]:
    count = count or args.gpu_count
    values = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if len(values) < count:
        raise ValueError(f"need {count} GPU ids, got {values}")
    return values[:count]


def _gpu_string(args, count: int | None = None) -> str:
    return ",".join(_gpu_ids(args, count))


def _run_task(args, task: Task, gpu_count: int, gpus: str, port: int, resume_test: bool = False) -> None:
    if _completed(task):
        _enrich_end_to_end_metrics(task, resume_test)
        return
    log = task.output / "launcher.log"
    if resume_test and not (task.output / "checkpoint-8").exists():
        _run_process(_train_command(args, task, gpu_count, port, stop_after=8), gpus, log)
    resume = task.output / "checkpoint-8" if resume_test else _latest_checkpoint(task)
    _run_process(_train_command(args, task, gpu_count, port, resume=resume), gpus, log)
    if not _completed(task):
        raise RuntimeError(f"task did not reach its declared final step: {task.output}")
    _enrich_end_to_end_metrics(task, resume_test)


def _enrich_end_to_end_metrics(task: Task, resumed: bool) -> None:
    metrics_path = task.output / "metrics.json"
    state_path = task.output / "trainer_state.json"
    log_path = task.output / "launcher.log"
    if not metrics_path.exists() or not state_path.exists() or not log_path.exists():
        return
    metrics = json.loads(metrics_path.read_text())
    state = json.loads(state_path.read_text())
    step_losses = [float(row["loss"]) for row in state.get("log_history", []) if "loss" in row and "step" in row]
    runtimes = [float(value) for value in re.findall(r"'train_runtime':\s*([0-9.]+)", log_path.read_text(errors="replace"))]
    required_segments = 2 if resumed else 1
    if len(step_losses) >= task.max_steps and len(runtimes) >= required_segments:
        runtime = sum(runtimes[-required_segments:])
        metrics["end_to_end_train_loss"] = sum(step_losses[-task.max_steps:]) / task.max_steps
        metrics["end_to_end_runtime"] = runtime
        metrics["end_to_end_samples_per_second"] = task.max_steps * 64 / runtime
        atomic_json(metrics_path, metrics)


def _finalize(task: Task) -> None:
    finalization = task.output / "finalization.json"
    if finalization.exists():
        state = json.loads(finalization.read_text())
        if state.get("status") == "verified" and state.get("recovery_deleted") is True:
            return
    subprocess.run([
        sys.executable, "-m", "rubric_dpo.cli.finalize_run",
        "--run-dir", str(task.output), "--final-step", str(task.max_steps), "--delete-recovery",
    ], check=True)


def _finalize_pilot(task: Task) -> None:
    """Keep trusted pilot metrics, not an unnecessary portable model."""
    verification_path = task.output / "pilot_verification.json"
    if verification_path.exists():
        verification = json.loads(verification_path.read_text())
        if verification.get("status") == "verified_metrics_only" and verification.get("recovery_deleted") is True:
            return
    if not _completed(task):
        raise RuntimeError(f"pilot is not complete: {task.output}")
    manifest_path = task.output / "run_manifest.json"
    metrics_path = task.output / "metrics.json"
    resolved_path = task.output / "resolved_config.json"
    metrics = json.loads(metrics_path.read_text())
    required = {
        "global_step": task.max_steps,
        "validation_nll": _metric(task, "baseline_loss_unweighted_dpo"),
        "validation_accuracy": _metric(task, "rewards/accuracies"),
    }
    if not all(math.isfinite(float(value)) for value in required.values()):
        raise FloatingPointError(f"pilot selection metrics are non-finite: {required}")
    checkpoints = sorted(task.output.glob("checkpoint-*"))
    for checkpoint in checkpoints:
        if not (checkpoint / "trusted_local_checkpoint.json").exists():
            raise RuntimeError(f"pilot checkpoint is not sealed: {checkpoint}")
    report = {
        "status": "verified_metrics_only",
        "purpose": "validation-only hyperparameter selection; portable pilot weights are not retained",
        "variant": task.variant,
        "seed": task.seed,
        "learning_rate": task.lr,
        "gamma": task.gamma,
        "alpha": task.alpha,
        **required,
        "metrics_sha256": sha256_file(metrics_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "resolved_config_sha256": sha256_file(resolved_path),
        "recovery_deleted": False,
    }
    atomic_json(verification_path, report)
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)
    stale_merged = task.output / "final_merged"
    if stale_merged.exists() and not any(stale_merged.iterdir()):
        stale_merged.rmdir()
    report["recovery_deleted"] = True
    atomic_json(verification_path, report)


def _metric(task: Task, suffix: str) -> float:
    metrics = json.loads((task.output / "metrics.json").read_text())
    keys = [key for key in metrics if key.endswith(suffix)]
    if not keys:
        raise KeyError(f"metric ending {suffix!r} missing in {task.output}: {sorted(metrics)}")
    value = float(metrics[keys[-1]])
    if not math.isfinite(value):
        raise FloatingPointError(f"non-finite metric {keys[-1]}={value}")
    return value


def run_smoke(args) -> None:
    prerequisites = {
        "preflight": args.root / "artifacts/preflight/preflight.json",
        "source_parity": args.root / "artifacts/baselines/qwen3_8b_ultrafeedback_margin_consumers_v1/source_parity.json",
        "reference_parity": args.reference_cache / "verification.json",
    }
    for name, path in prerequisites.items():
        if not path.exists():
            raise FileNotFoundError(f"smoke prerequisite {name} missing: {path}")
        state = json.loads(path.read_text())
        expected = "complete" if name == "preflight" else "passed"
        if state.get("status") != expected:
            raise RuntimeError(f"smoke prerequisite {name} is not {expected}: {path}")
    tasks = [Task(v, args.output_root / "smoke" / v, 1e-6) for v in VARIANTS]
    for index, task in enumerate(tasks):
        _run_task(args, task, args.gpu_count, _gpu_string(args), 29500 + index, resume_test=task.variant == "dpo")
    report = {"status": "partially_verified", "tasks": []}
    for task in tasks:
        metrics = json.loads((task.output / "metrics.json").read_text())
        resolved = json.loads((task.output / "resolved_config.json").read_text())
        run_manifest = json.loads((task.output / "run_manifest.json").read_text())
        finalization_path = task.output / "finalization.json"
        finalized = (
            finalization_path.exists()
            and json.loads(finalization_path.read_text()).get("status") == "verified"
        )
        passed = (
            _completed(task)
            and all(math.isfinite(float(value)) for value in metrics.values() if isinstance(value, (int, float)))
            and resolved["effective_batch_size"] == 64
            and run_manifest.get("ref_model_is_none") is True
            and ((task.output / "checkpoint-16/pytorch_model_fsdp_0").exists() or finalized)
            and (task.variant != "dpo" or (task.output / "checkpoint-8").exists() or finalized)
        )
        report["tasks"].append({"variant": task.variant, "passed": passed, "pair_id_root": resolved["pair_id_root"], "metrics": metrics})
    report["shared_pair_order"] = len({row["pair_id_root"] for row in report["tasks"]}) == 1
    report["status"] = "verified-for-controlled-UF" if report["shared_pair_order"] and all(row["passed"] for row in report["tasks"]) else "failed"
    atomic_json(args.output_root / "smoke/smoke_verification.json", report)
    if report["status"] == "verified-for-controlled-UF":
        for task in tasks:
            _finalize(task)


def run_probe(args) -> None:
    if args.gpu_count != 4:
        raise ValueError("the 2x2 parallel probe requires --gpu-count 4")
    report_path = args.output_root / "parallel_probe.json"
    if report_path.exists() and not args.force_probe:
        report = json.loads(report_path.read_text())
        probe_manifest_path = args.output_root / "probe_2gpu/dpo/run_manifest.json"
        if report.get("status") == "failed" and probe_manifest_path.exists():
            probe_manifest = json.loads(probe_manifest_path.read_text())
            if probe_manifest.get("status") == "running":
                atomic_json(probe_manifest_path, {
                    **probe_manifest,
                    "status": "failed_resource_gate",
                    "failure_reason": report.get("reason", "two-GPU safety probe failed"),
                    "fallback": "four_gpu_sequential",
                })
        print(json.dumps(report, indent=2))
        return
    four = Task("dpo", args.output_root / "smoke/dpo", 1e-6)
    if not _completed(four):
        raise RuntimeError("run the four-GPU smoke before the two-GPU probe")
    two = Task("dpo", args.output_root / "probe_2gpu/dpo", 1e-6)
    try:
        _run_task(args, two, 2, "0,1", 29600, resume_test=True)
    except Exception as error:
        atomic_json(report_path, {
            "status": "failed", "reason": f"{type(error).__name__}: {error}",
            "fallback": "four_gpu_sequential", "research_configuration_changed": False,
        })
        return
    four_metrics = json.loads((four.output / "metrics.json").read_text())
    two_metrics = json.loads((two.output / "metrics.json").read_text())
    peak = float(two_metrics["peak_reserved_gb"])
    throughput_ratio = 2.0 * float(two_metrics["end_to_end_samples_per_second"]) / float(four_metrics["end_to_end_samples_per_second"])
    passed = peak <= 65.0 and throughput_ratio >= 0.9
    atomic_json(report_path, {
        "status": "passed" if passed else "failed", "peak_reserved_gb": peak,
        "two_job_combined_to_four_gpu_throughput": throughput_ratio,
        "requires": {"peak_reserved_gb_lte": 65.0, "throughput_ratio_gte": 0.9},
    })
    _finalize(two)


def _selection_key(task: Task, parameter: float) -> tuple[float, float, float]:
    return (
        _metric(task, "baseline_loss_unweighted_dpo"),
        -_metric(task, "rewards/accuracies"),
        parameter,
    )


def run_pilots(args) -> None:
    pilot_root = args.output_root / "pilots"
    dpo_tasks = [Task("dpo", pilot_root / f"dpo_lr_{lr:.1e}", lr, max_steps=128, save_steps=128, evaluate=True) for lr in (5e-7, 1e-6)]
    for index, task in enumerate(dpo_tasks):
        atomic_json(args.output_root / "pilot_progress.json", {
            "status": "running", "phase": "dpo_lr", "current_run": str(task.output),
            "completed_runs": index, "total_runs": 8,
        })
        _run_task(args, task, args.gpu_count, _gpu_string(args), 29700 + index)
        _finalize_pilot(task)
    selected_dpo = min(dpo_tasks, key=lambda task: _selection_key(task, task.lr))
    lr = selected_dpo.lr
    gamma_tasks = [Task("mmpo", pilot_root / f"mmpo_gamma_{value:g}", lr, gamma=value, max_steps=128, save_steps=128, evaluate=True) for value in (1.0, 2.2, 4.0)]
    alpha_tasks = [Task("odpo_loggap", pilot_root / f"odpo_alpha_{value:g}", lr, alpha=value, max_steps=128, save_steps=128, evaluate=True) for value in (0.1, 0.5, 1.0)]
    for index, task in enumerate(gamma_tasks + alpha_tasks):
        atomic_json(args.output_root / "pilot_progress.json", {
            "status": "running", "phase": "mmpo_gamma_or_odpo_alpha", "current_run": str(task.output),
            "completed_runs": 2 + index, "total_runs": 8,
        })
        _run_task(args, task, args.gpu_count, _gpu_string(args), 29710 + index)
        _finalize_pilot(task)
    gamma_task = min(gamma_tasks, key=lambda task: _selection_key(task, task.gamma))
    alpha_task = min(alpha_tasks, key=lambda task: _selection_key(task, task.alpha))
    selection = {
        "status": "frozen_before_test", "learning_rate": lr,
        "gamma": gamma_task.gamma, "alpha": alpha_task.alpha,
        "selection_metric": "validation hard-preference NLL; accuracy then smaller parameter",
        "selected_runs": {"dpo": str(selected_dpo.output), "mmpo": str(gamma_task.output), "odpo": str(alpha_task.output)},
    }
    atomic_json(args.output_root / "selection.json", selection)
    atomic_json(args.output_root / "pilot_progress.json", {
        "status": "complete", "completed_runs": 8, "total_runs": 8,
        "selection": selection,
    })


def _run_parallel_pair(args, left: Task, right: Task, port: int) -> None:
    selected = _gpu_ids(args, 4)
    commands = [
        (_train_command(args, left, 2, port, resume=_latest_checkpoint(left)), ",".join(selected[:2]), left),
        (_train_command(args, right, 2, port + 1, resume=_latest_checkpoint(right)), ",".join(selected[2:]), right),
    ]
    processes = []
    for index, (command, gpus, task) in enumerate(commands):
        if _completed(task):
            continue
        task.output.mkdir(parents=True, exist_ok=True)
        log = (task.output / "launcher.log").open("a", encoding="utf-8")
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpus
        processes.append((subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log, task))
        if index == 0:
            time.sleep(20)
    failures = []
    for process, log, task in processes:
        code = process.wait(); log.close()
        if code or not _completed(task):
            failures.append((task.output, code))
    if failures:
        raise RuntimeError(f"parallel tasks failed: {failures}")


def run_full(args) -> None:
    selection_path = args.output_root / "selection.json"
    smoke_path = args.output_root / "smoke/smoke_verification.json"
    if not selection_path.exists() or not smoke_path.exists():
        raise RuntimeError("smoke gate and pilot selection must exist before full runs")
    if json.loads(smoke_path.read_text()).get("status") != "verified-for-controlled-UF":
        raise RuntimeError("baseline smoke gate is not verified")
    selection = json.loads(selection_path.read_text())
    tasks = []
    for seed in (13, 42, 100):
        for variant in VARIANTS:
            tasks.append(Task(
                variant, args.output_root / "full" / variant / f"seed_{seed}",
                selection["learning_rate"], gamma=selection["gamma"], alpha=selection["alpha"],
                seed=seed, max_steps=566, save_steps=283, evaluate=True, eval_split="test",
            ))
    probe = args.output_root / "parallel_probe.json"
    parallel = args.gpu_count == 4 and args.parallel == "auto" and probe.exists() and json.loads(probe.read_text()).get("status") == "passed"
    if parallel:
        for wave, start in enumerate(range(0, len(tasks), 2)):
            pair = tasks[start:start + 2]
            if len(pair) == 2:
                _run_parallel_pair(args, pair[0], pair[1], 29800 + 2 * wave)
            else:
                _run_task(args, pair[0], 2, "0,1", 29800 + 2 * wave)
            for task in pair:
                _finalize(task)
    else:
        for index, task in enumerate(tasks):
            _run_task(args, task, args.gpu_count, _gpu_string(args), 29900 + index)
            _finalize(task)


def status(args) -> None:
    rows = []
    for path in sorted(args.output_root.glob("**/run_manifest.json")):
        data = json.loads(path.read_text())
        rows.append({"run": str(path.parent.relative_to(args.output_root)), "status": data.get("status"), "step": data.get("global_step")})
    print(json.dumps(rows, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate-aware four-GPU baseline scheduler")
    parser.add_argument("phase", choices=("smoke", "probe", "pilots", "full", "status"))
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/rubric"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--parallel", choices=("auto", "off"), default="auto")
    parser.add_argument("--gpu-count", type=int, choices=(2, 4), default=4)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--two-gpu-offload", action="store_true")
    parser.add_argument("--force-probe", action="store_true", help="repeat an existing two-GPU safety probe")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.model = args.model or args.root / "models/qwen3-8b"
    args.dataset_dir = args.dataset_dir or args.root / "data/processed/ultrafeedback/v2/tokens_qwen3_8b_non_thinking"
    args.reference_cache = args.reference_cache or args.root / "data/cache/ultrafeedback_qwen3_8b_ref_v1"
    args.output_root = args.output_root or args.root / "runs/baselines/qwen3_8b_ultrafeedback_margin_consumers_v1"
    if args.gpu_count == 2 and args.phase in {"smoke", "pilots", "full"} and not args.two_gpu_offload:
        parser.error("two-GPU full-parameter training requires --two-gpu-offload and a successful smoke")
    {"smoke": run_smoke, "probe": run_probe, "pilots": run_pilots, "full": run_full, "status": status}[args.phase](args)


if __name__ == "__main__":
    main()
