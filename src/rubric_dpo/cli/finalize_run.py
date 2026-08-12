from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rubric_dpo.utils import atomic_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify, merge, and optionally remove recovery shards")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--final-step", type=int, required=True)
    parser.add_argument("--delete-recovery", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics.json"
    if not manifest_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("run manifest and metrics must exist before finalization")
    manifest = json.loads(manifest_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    if manifest.get("status") != "complete" or int(metrics.get("global_step", -1)) != args.final_step:
        raise ValueError("run is not a complete final-step checkpoint")
    checkpoint = run_dir / f"checkpoint-{args.final_step}"
    fsdp_dir = checkpoint / "pytorch_model_fsdp_0"
    if not fsdp_dir.exists():
        raise FileNotFoundError(f"FSDP checkpoint missing: {fsdp_dir}")
    merged = run_dir / "final_merged"
    working = run_dir / ".final_merged.tmp"
    finalize_log = run_dir / "finalize.log"
    failure_path = run_dir / "finalization_failure.json"
    merged_weight_bytes_before_cast = None
    try:
        existing_weights = sorted(merged.glob("*.safetensors")) if merged.exists() else []
        if not existing_weights:
            if working.exists():
                shutil.rmtree(working)
            command = [
                str(Path(sys.executable).with_name("accelerate")), "merge-weights", str(fsdp_dir), str(working),
            ]
            with finalize_log.open("a", encoding="utf-8") as log:
                log.write("COMMAND " + " ".join(command) + "\n")
                log.flush()
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            weight_files = sorted(working.glob("*.safetensors"))
            if not weight_files:
                raise FileNotFoundError("temporary merged model contains no safetensors")
            merged_weight_bytes = sum(path.stat().st_size for path in weight_files)
            merged_weight_bytes_before_cast = merged_weight_bytes
            if merged_weight_bytes < 14 * 1024**3:
                raise ValueError(f"merged Qwen3 weights are unexpectedly small: {merged_weight_bytes} bytes")
            observed_dtypes = set()
            for path in weight_files:
                with safe_open(path, framework="pt", device="cpu") as handle:
                    observed_dtypes.update(handle.get_tensor(key).dtype for key in handle.keys())
            if torch.float32 in observed_dtypes:
                for path in weight_files:
                    with safe_open(path, framework="pt", device="cpu") as handle:
                        metadata = handle.metadata()
                        tensors = {
                            key: (handle.get_tensor(key).to(torch.bfloat16) if handle.get_tensor(key).is_floating_point() else handle.get_tensor(key))
                            for key in handle.keys()
                        }
                    temporary = path.with_name(f".{path.name}.bf16.tmp")
                    save_file(tensors, temporary, metadata=metadata)
                    temporary.replace(path)
                    del tensors
            model_path = Path(manifest["model_path"])
            for name in (
                "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "merges.txt", "vocab.json", "chat_template.jinja",
            ):
                source = model_path / name
                if source.exists():
                    shutil.copy2(source, working / name)
            if merged.exists():
                if any(merged.iterdir()):
                    quarantine = run_dir / "final_merged.incomplete"
                    if quarantine.exists():
                        raise FileExistsError(f"cannot quarantine incomplete merged model: {quarantine}")
                    merged.rename(quarantine)
                else:
                    merged.rmdir()
            working.replace(merged)
        weight_files = sorted(merged.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError("merged model contains no safetensors")
        merged_weight_bytes = sum(path.stat().st_size for path in weight_files)
        total_weight_bytes = merged_weight_bytes
        if merged_weight_bytes_before_cast is None:
            merged_weight_bytes_before_cast = total_weight_bytes
        if not 14 * 1024**3 <= total_weight_bytes <= 20 * 1024**3:
            raise ValueError(f"portable BF16 Qwen3 size outside expected range: {total_weight_bytes} bytes")
        checksums = {path.name: sha256_file(path) for path in weight_files}
    except Exception as error:
        atomic_json(failure_path, {
            "status": "failed", "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(), "log": str(finalize_log),
            "checkpoint_preserved": fsdp_dir.exists(),
        })
        raise
    finalization = {
        "status": "verified", "final_step": args.final_step, "merged_model": str(merged),
        "model_checksums": checksums, "metrics_sha256": sha256_file(metrics_path),
        "merged_weight_bytes_before_cast": merged_weight_bytes_before_cast,
        "total_weight_bytes": total_weight_bytes, "portable_dtype": "bfloat16",
        "recovery_deleted": False,
    }
    atomic_json(run_dir / "finalization.json", finalization)
    if failure_path.exists():
        failure_path.unlink()
    if args.delete_recovery:
        for path in sorted(run_dir.glob("checkpoint-*")):
            if path.is_dir():
                shutil.rmtree(path)
        if any(path.is_dir() for path in run_dir.glob("checkpoint-*")):
            raise RuntimeError("recovery cleanup did not remove every checkpoint directory")
        finalization["recovery_deleted"] = True
        atomic_json(run_dir / "finalization.json", finalization)
    print(json.dumps(json.loads((run_dir / "finalization.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
