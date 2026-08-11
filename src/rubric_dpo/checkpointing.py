from __future__ import annotations

from pathlib import Path

from .utils import atomic_json, sha256_file

SMALL_TRUSTED_PATTERNS = (
    "scheduler.pt",
    "trainer_state.json",
    "rng_state_*.pth",
    "optimizer_0/.metadata",
    "pytorch_model_fsdp_0/.metadata",
)


def _trusted_files(checkpoint: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in SMALL_TRUSTED_PATTERNS:
        files.extend(checkpoint.glob(pattern))
    return sorted(set(files))


def seal_local_checkpoint(checkpoint: Path, output_dir: Path) -> dict:
    checkpoint = checkpoint.resolve()
    output_dir = output_dir.resolve()
    if checkpoint.parent != output_dir or not checkpoint.name.startswith("checkpoint-"):
        raise ValueError("checkpoint must be a direct checkpoint-N child of this run output")
    files = _trusted_files(checkpoint)
    if not files or not (checkpoint / "scheduler.pt").exists():
        raise FileNotFoundError("checkpoint is incomplete and cannot be sealed")
    for path in files:
        if path.is_symlink():
            raise ValueError(f"trusted checkpoint file may not be a symlink: {path}")
    manifest = {
        "status": "trusted_local_checkpoint",
        "checkpoint": checkpoint.name,
        "files": {str(path.relative_to(checkpoint)): sha256_file(path) for path in files},
    }
    atomic_json(checkpoint / "trusted_local_checkpoint.json", manifest)
    return manifest


def validate_local_checkpoint(checkpoint: Path, output_dir: Path) -> dict:
    import json

    checkpoint = checkpoint.resolve()
    output_dir = output_dir.resolve()
    if checkpoint.parent != output_dir or not checkpoint.name.startswith("checkpoint-"):
        raise ValueError("refusing to load a checkpoint outside the current run output")
    trust_path = checkpoint / "trusted_local_checkpoint.json"
    if trust_path.is_symlink() or not trust_path.exists():
        raise FileNotFoundError(f"trusted-local seal missing: {trust_path}")
    manifest = json.loads(trust_path.read_text())
    if manifest.get("status") != "trusted_local_checkpoint" or manifest.get("checkpoint") != checkpoint.name:
        raise ValueError("invalid trusted-local checkpoint manifest")
    for relative, expected in manifest["files"].items():
        path = checkpoint / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"trusted checkpoint checksum failed: {relative}")
        if path.name in {"scheduler.pt"} or path.name.startswith("rng_state_"):
            if path.stat().st_size > 4 * 1024 * 1024:
                raise ValueError(f"unexpectedly large pickle-bearing checkpoint file: {relative}")
    return manifest
