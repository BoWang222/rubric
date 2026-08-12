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
from rubric_dpo.checkpointing import validate_local_checkpoint
from rubric_dpo.utils import atomic_json, sha256_file

PILOT_STEPS = 64
FIXED_MMPO_GAMMA = 2.2


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("--seeds must be a comma-separated list of integers") from error
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    if any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("--seeds must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("--seeds must not contain duplicates")
    return seeds


def _parse_variants(value: str) -> tuple[str, ...]:
    variants = tuple(item.strip() for item in value.split(",") if item.strip())
    if not variants:
        raise argparse.ArgumentTypeError("--variants must contain at least one baseline variant")
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown baseline variants: {sorted(unknown)}")
    if len(set(variants)) != len(variants):
        raise argparse.ArgumentTypeError("--variants must not contain duplicates")
    return variants


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
    save_checkpoints: bool = True
    evaluate: bool = False
    eval_split: str = "validation"


def _train_command(args, task: Task, gpu_count: int, port: int, stop_after: int | None = None, resume: Path | None = None) -> list[str]:
    config_name = "fsdp_2gpu_cpu_offload.yaml" if gpu_count == 2 and args.two_gpu_offload else f"fsdp_{gpu_count}gpu.yaml"
    fsdp = args.root / "configs/accelerate" / config_name
    batch_contract = {
        2: (1, 32),
        4: (2, 8),
        8: (2, 4),
    }
    per_device, accumulation = batch_contract[gpu_count]
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
        "--per-device-batch-size", str(per_device),
        "--gradient-accumulation-steps", str(accumulation),
        "--logging-steps", "1" if task.max_steps <= 16 else "10",
    ]
    if task.save_checkpoints:
        command += ["--save-steps", str(task.save_steps)]
    else:
        command += ["--no-save"]
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
            try:
                validate_local_checkpoint(path, task.output)
            except (FileNotFoundError, ValueError):
                continue
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _run_process(command: list[str], gpus: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("NCCL_DEBUG", "WARN")
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
    try:
        checkpoint8 = task.output / "checkpoint-8"
        if resume_test:
            try:
                validate_local_checkpoint(checkpoint8, task.output)
            except (FileNotFoundError, ValueError):
                _run_process(_train_command(args, task, gpu_count, port, stop_after=8), gpus, log)
        resume = checkpoint8 if resume_test else _latest_checkpoint(task)
        _run_process(_train_command(args, task, gpu_count, port, resume=resume), gpus, log)
    except Exception as error:
        _record_task_failure(task, error, log)
        raise
    if not _completed(task):
        error = RuntimeError(f"task did not reach its declared final step: {task.output}")
        _record_task_failure(task, error, log)
        raise error
    _enrich_end_to_end_metrics(task, resume_test)


def _record_task_failure(task: Task, error: Exception, log_path: Path) -> None:
    manifest_path = task.output / "run_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {}
    incomplete = []
    trusted = []
    for checkpoint in sorted(task.output.glob("checkpoint-*")):
        if not checkpoint.is_dir():
            continue
        try:
            validate_local_checkpoint(checkpoint, task.output)
            trusted.append(checkpoint.name)
        except (FileNotFoundError, ValueError):
            incomplete.append(checkpoint.name)
    atomic_json(manifest_path, {
        **manifest,
        "status": "failed",
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "launcher_log": str(log_path),
        "trusted_checkpoints": trusted,
        "incomplete_checkpoints": incomplete,
    })


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
        recovery_dirs = [path for path in task.output.glob("checkpoint-*") if path.is_dir()]
        if state.get("status") == "verified" and state.get("recovery_deleted") is True and not recovery_dirs:
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
    dpo_tasks = [Task("dpo", pilot_root / f"dpo_lr_{lr:.1e}", lr, max_steps=PILOT_STEPS, save_checkpoints=False, evaluate=True) for lr in (5e-7, 1e-6)]
    for index, task in enumerate(dpo_tasks):
        atomic_json(args.output_root / "pilot_progress.json", {
            "status": "running", "phase": "dpo_lr", "current_run": str(task.output),
            "completed_runs": index, "total_runs": 5,
        })
        _run_task(args, task, args.gpu_count, _gpu_string(args), 29700 + index)
        _finalize_pilot(task)
    selected_dpo = min(dpo_tasks, key=lambda task: _selection_key(task, task.lr))
    lr = selected_dpo.lr
    alpha_tasks = [Task("odpo_loggap", pilot_root / f"odpo_alpha_{value:g}", lr, alpha=value, max_steps=PILOT_STEPS, save_checkpoints=False, evaluate=True) for value in (0.1, 0.5, 1.0)]
    for index, task in enumerate(alpha_tasks):
        atomic_json(args.output_root / "pilot_progress.json", {
            "status": "running", "phase": "odpo_alpha", "current_run": str(task.output),
            "completed_runs": 2 + index, "total_runs": 5,
        })
        _run_task(args, task, args.gpu_count, _gpu_string(args), 29710 + index)
        _finalize_pilot(task)
    alpha_task = min(alpha_tasks, key=lambda task: _selection_key(task, task.alpha))
    selection = {
        "status": "frozen_before_test", "learning_rate": lr,
        "gamma": FIXED_MMPO_GAMMA, "alpha": alpha_task.alpha,
        "selection_metric": "validation hard-preference NLL; accuracy then smaller parameter",
        "fixed_hyperparameters": {
            "mmpo_gamma": FIXED_MMPO_GAMMA,
            "source": "MMPO author 8B UltraFeedback recipe",
            "selected_by_local_pilot": False,
        },
        "selected_runs": {"dpo": str(selected_dpo.output), "mmpo": None, "odpo": str(alpha_task.output)},
    }
    atomic_json(args.output_root / "selection.json", selection)
    atomic_json(args.output_root / "pilot_progress.json", {
        "status": "complete", "completed_runs": 5, "total_runs": 5,
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
    for seed in args.seeds:
        for variant in args.variants:
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
    parser = argparse.ArgumentParser(description="Gate-aware multi-GPU baseline scheduler")
    parser.add_argument("phase", choices=("smoke", "probe", "pilots", "full", "status"))
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/rubric"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--parallel", choices=("auto", "off"), default="auto")
    parser.add_argument("--gpu-count", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seeds", type=_parse_seeds, default=(13, 42, 100), help="comma-separated full-run seeds")
    parser.add_argument("--variants", type=_parse_variants, default=VARIANTS, help="comma-separated full-run variants")
    parser.add_argument("--two-gpu-offload", action="store_true")
    parser.add_argument("--two-gpu-no-offload", action="store_true")
    parser.add_argument("--force-probe", action="store_true", help="repeat an existing two-GPU safety probe")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.model = args.model or args.root / "models/qwen3-8b"
    args.dataset_dir = args.dataset_dir or args.root / "data/processed/ultrafeedback/v2/tokens_qwen3_8b_non_thinking"
    args.reference_cache = args.reference_cache or args.root / "data/cache/ultrafeedback_qwen3_8b_ref_v1"
    args.output_root = args.output_root or args.root / "runs/baselines/qwen3_8b_ultrafeedback_margin_consumers_v1"
    if args.two_gpu_offload and args.two_gpu_no_offload:
        parser.error("choose exactly one two-GPU memory policy")
    if args.gpu_count == 2 and args.phase in {"smoke", "pilots", "full"}:
        if not (args.two_gpu_offload or args.two_gpu_no_offload):
            parser.error("two-GPU training requires an explicit --two-gpu-offload or --two-gpu-no-offload policy")
    {"smoke": run_smoke, "probe": run_probe, "pilots": run_pilots, "full": run_full, "status": status}[args.phase](args)


if __name__ == "__main__":
    main()
