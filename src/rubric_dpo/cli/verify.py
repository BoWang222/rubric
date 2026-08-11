from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig

from rubric_dpo.collator import BaselinePreferenceCollator
from rubric_dpo.checkpointing import seal_local_checkpoint
from rubric_dpo.constants import QWEN_PAD_ID
from rubric_dpo.trainer import BaselineDPOTrainer
from rubric_dpo.utils import atomic_json


def source_parity_main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen FP64 source/formula parity gates")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_losses.py", "tests/test_collator.py", "tests/test_data_contract.py"],
        cwd=args.project_root, text=True, capture_output=True,
    )
    report = {"status": "passed" if result.returncode == 0 else "failed", "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


def reference_cache_main() -> None:
    parser = argparse.ArgumentParser(description="Compare durable reference logps to TRL concatenated forward")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    rows = pq.read_table(args.dataset_dir / "smoke_1024.parquet").slice(0, args.samples).to_pylist()
    cache = {row["pair_id"]: row for row in pq.read_table(args.reference_cache / "reference_logps.parquet").to_pylist()}
    joined = [{**row, "ref_chosen_logps": cache[row["pair_id"]]["ref_chosen_logps"], "ref_rejected_logps": cache[row["pair_id"]]["ref_rejected_logps"]} for row in rows]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id != QWEN_PAD_ID:
        raise AssertionError("unexpected Qwen pad token")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", use_cache=False,
    ).cuda().eval()
    config = DPOConfig(
        output_dir=tempfile.mkdtemp(prefix="rubric-cache-parity-"), beta=.01,
        precompute_ref_log_probs=True, max_length=2048, max_prompt_length=1024,
        max_completion_length=1024, truncation_mode="keep_end", use_logits_to_keep=True,
        bf16=True, remove_unused_columns=False, report_to=[],
    )
    dataset = Dataset.from_list(joined)
    collator = BaselinePreferenceCollator(pad_token_id=tokenizer.pad_token_id)
    trainer = BaselineDPOTrainer(
        model=model, ref_model=None, args=config, data_collator=collator,
        train_dataset=dataset, processing_class=tokenizer, baseline_variant="dpo",
    )
    batch = trainer._prepare_inputs(collator(joined))
    with torch.inference_mode():
        output = trainer.concatenated_forward(trainer.model, batch)
    cached_chosen = batch["ref_chosen_logps"]
    cached_rejected = batch["ref_rejected_logps"]
    chosen_error = (output["chosen_logps"] - cached_chosen).abs().detach().cpu()
    rejected_error = (output["rejected_logps"] - cached_rejected).abs().detach().cpu()
    max_error = float(torch.cat([chosen_error, rejected_error]).max())
    report = {
        "status": "passed" if max_error <= args.atol else "failed", "samples": args.samples,
        "atol": args.atol, "max_abs_error": max_error,
        "chosen_abs_errors": chosen_error.tolist(), "rejected_abs_errors": rejected_error.tolist(),
        "ref_model_is_none": trainer.ref_model is None,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


def seal_checkpoint_main() -> None:
    parser = argparse.ArgumentParser(description="Seal a checkpoint produced locally by this run before trusted resume")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = seal_local_checkpoint(args.checkpoint, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    raise SystemExit("Use rubric-verify-source-parity or rubric-verify-reference-cache")
