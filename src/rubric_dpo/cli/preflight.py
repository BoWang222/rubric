from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from rubric_dpo.constants import DPO_COMMIT, MMPO_COMMIT, MODEL_REVISION, ODPO_COMMIT, TRL_COMMIT
from rubric_dpo.utils import atomic_json, sha256_file


def _run(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()


def _git_commit(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    return _run(["git", "-C", str(path), "rev-parse", "HEAD"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-fast environment, data and provenance audit")
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/rubric"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-gpus", type=int)
    parser.add_argument("--min-gpus", type=int, default=2)
    parser.add_argument("--allowed-cuda", default="12.1,12.4")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "artifacts/preflight/preflight.json"
    required_paths = {
        "model": root / "models/qwen3-8b",
        "raw_ultrafeedback": root / "data/raw/ultrafeedback_raw",
        "h4_ultrafeedback": root / "data/raw/ultrafeedback_binarized",
        "trl": root / "refs/trl",
        "dpo": root / "refs/dpo",
        "mmpo": root / "refs/mmpo",
        "odpo": root / "refs/odpo",
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"preflight paths missing: {missing}")
    import accelerate, datasets, deepspeed, flash_attn, huggingface_hub, transformers, triton, trl
    versions = {
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "trl": trl.__version__, "accelerate": accelerate.__version__,
        "datasets": datasets.__version__, "deepspeed": deepspeed.__version__, "flash_attn": flash_attn.__version__,
        "huggingface_hub": huggingface_hub.__version__, "triton": triton.__version__,
    }
    expected = {"transformers": "4.52.4", "trl": "0.19.1", "accelerate": "1.8.1"}
    for name, value in expected.items():
        if versions[name] != value:
            raise AssertionError(f"{name} expected {value}, got {versions[name]}")
    if not torch.__version__.startswith("2.5.1+"):
        raise AssertionError(f"torch expected a CUDA build of 2.5.1, got {torch.__version__}")
    allowed_cuda = {value.strip() for value in args.allowed_cuda.split(",") if value.strip()}
    if torch.version.cuda not in allowed_cuda or torch._C._GLIBCXX_USE_CXX11_ABI is not False:
        raise AssertionError("Torch CUDA/ABI contract changed")
    gpu_count = torch.cuda.device_count()
    if args.expected_gpus is not None and gpu_count != args.expected_gpus:
        raise AssertionError(f"expected {args.expected_gpus} visible GPUs, got {gpu_count}")
    if gpu_count < args.min_gpus:
        raise AssertionError(f"expected at least {args.min_gpus} visible GPUs, got {gpu_count}")
    pip_check = _run([sys.executable, "-m", "pip", "check"])
    gpu_csv = _run([
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]).splitlines()
    shards = sorted((root / "models/qwen3-8b").glob("model-*-of-*.safetensors"))
    if len(shards) != 5:
        raise AssertionError(f"expected five Qwen3 shards, got {len(shards)}")
    disk = shutil.disk_usage(root)
    report = {
        "status": "complete", "root": str(root), "versions": versions, "pip_check": pip_check,
        "gpus": gpu_csv, "gpu_count": torch.cuda.device_count(),
        "data_disk_free_gb": disk.free / 1024**3,
        "model_revision": MODEL_REVISION,
        "model_shards": {path.name: sha256_file(path) for path in shards},
        "source_commits": {name: _git_commit(path) for name, path in required_paths.items() if name in {"trl", "dpo", "mmpo", "odpo"}},
        "environment_deviation": "Reused the already verified rubric conda environment instead of rebuilding the CUDA stack under uv.",
    }
    expected_commits = {"trl": TRL_COMMIT, "dpo": DPO_COMMIT, "mmpo": MMPO_COMMIT, "odpo": ODPO_COMMIT}
    for name, expected_commit in expected_commits.items():
        if report["source_commits"].get(name) != expected_commit:
            raise AssertionError(f"{name} source commit changed: expected {expected_commit}, got {report['source_commits'].get(name)}")
    atomic_json(output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
