from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig

from rubric_dpo.collator import BaselinePreferenceCollator
from rubric_dpo.constants import BASELINE_ID, MODEL_REVISION, QWEN_PAD_ID, TRL_COMMIT, VARIANTS
from rubric_dpo.trainer import BaselineDPOTrainer
from rubric_dpo.utils import atomic_json, canonical_json, ordered_id_root, seed_everything, sha256_file, sha256_text


class StopAfterStepCallback(TrainerCallback):
    def __init__(self, step: int):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.step:
            control.should_save = True
            control.should_training_stop = True
        return control


def _load_dataset(path: Path, cache_path: Path, split: str) -> Dataset:
    rows = pq.read_table(path).to_pylist()
    cache_rows = [row for row in pq.read_table(cache_path).to_pylist() if row["split"] == split]
    cache = {row["pair_id"]: row for row in cache_rows}
    if len(cache) != len(cache_rows):
        raise ValueError("reference cache contains duplicate pair IDs")
    ids = [row["pair_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset contains duplicate pair IDs")
    missing = set(ids) - set(cache)
    if missing:
        raise ValueError(f"reference cache misses {len(missing)} pairs")
    joined = [
        {
            **row,
            "ref_chosen_logps": float(cache[row["pair_id"]]["ref_chosen_logps"]),
            "ref_rejected_logps": float(cache[row["pair_id"]]["ref_rejected_logps"]),
        }
        for row in rows
    ]
    return Dataset.from_list(joined)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one controlled UltraFeedback baseline")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-split", choices=("smoke_1024", "train"), default="train")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=566)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=283)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--evaluate-after-train", action="store_true")
    parser.add_argument("--eval-split", choices=("validation", "test"), default="validation")
    parser.add_argument("--eval-max-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    train_path = args.dataset_dir / f"{args.train_split}.parquet"
    cache_path = args.reference_cache / "reference_logps.parquet"
    cache_manifest = json.loads((args.reference_cache / "manifest.json").read_text())
    token_manifest = json.loads((args.dataset_dir / "tokenization.json").read_text())
    if cache_manifest["tokenization_root_digest"] != token_manifest["root_digest"]:
        raise ValueError("reference cache/tokenized dataset manifest mismatch")
    train_dataset = _load_dataset(train_path, cache_path, "train")
    eval_dataset = None
    if args.evaluate_after_train:
        eval_dataset = _load_dataset(args.dataset_dir / f"{args.eval_split}.parquet", cache_path, args.eval_split)
        if args.eval_max_samples is not None:
            eval_dataset = eval_dataset.select(range(min(args.eval_max_samples, len(eval_dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id != QWEN_PAD_ID:
        raise AssertionError("unexpected Qwen3 pad token")
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", use_cache=False,
    )
    model.config.use_cache = False
    training_args = DPOConfig(
        output_dir=str(args.output_dir),
        beta=args.beta,
        loss_type="sigmoid",
        label_smoothing=0.0,
        precompute_ref_log_probs=True,
        max_length=2048,
        max_prompt_length=1024,
        max_completion_length=1024,
        truncation_mode="keep_end",
        use_logits_to_keep=True,
        padding_free=False,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=1.0,
        max_steps=args.max_steps,
        optim="adamw_torch_fused",
        weight_decay=0.0,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.3,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_drop_last=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        save_only_model=False,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        load_best_model_at_end=False,
    )
    callbacks = [StopAfterStepCallback(args.stop_after_step)] if args.stop_after_step else []
    trainer = BaselineDPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        data_collator=BaselinePreferenceCollator(pad_token_id=tokenizer.pad_token_id),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
        baseline_variant=args.variant,
        gamma=args.gamma,
        alpha=args.alpha,
    )
    if trainer.is_world_process_zero():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parents[3]
        source_files = sorted(project_root.glob("src/rubric_dpo/**/*.py")) + sorted(project_root.glob("configs/**/*.yaml"))
        source_hashes = {str(path.relative_to(project_root)): sha256_file(path) for path in source_files}
        code_root_digest = sha256_text(canonical_json(source_hashes))
        resolved = {
            "baseline_id": BASELINE_ID,
            "variant": args.variant,
            "model_path": str(args.model.resolve()),
            "model_revision": MODEL_REVISION,
            "trl_commit": TRL_COMMIT,
            "dataset_tokenization_root": token_manifest["root_digest"],
            "reference_cache_root": cache_manifest["root_digest"],
            "pair_id_root": ordered_id_root(train_dataset["pair_id"]),
            "train_rows": len(train_dataset),
            "dropped_tail_pairs_per_epoch": len(train_dataset) % 64 if args.train_split == "train" else 0,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "gamma": args.gamma,
            "alpha": args.alpha,
            "effective_batch_size": args.per_device_batch_size * args.gradient_accumulation_steps * trainer.accelerator.num_processes,
            "max_steps": args.max_steps,
            "command": sys.argv,
            "code_root_digest": code_root_digest,
            "source_hashes": source_hashes,
            "tokenizer_chat_template_sha256": token_manifest["tokenizer_chat_template_sha256"],
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "source_identity": "source/formula-aligned controlled reproduction on Qwen3-8B + UltraFeedback",
        }
        atomic_json(args.output_dir / "resolved_config.json", resolved)
        atomic_json(args.output_dir / "run_manifest.json", {**resolved, "status": "running"})
    train_result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    metrics = dict(train_result.metrics)
    peak_reserved = torch.tensor(torch.cuda.max_memory_reserved() / (1024 ** 3), device=trainer.args.device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(peak_reserved, op=dist.ReduceOp.MAX)
    metrics["peak_reserved_gb"] = float(peak_reserved.item())
    metrics["global_step"] = trainer.state.global_step
    if args.evaluate_after_train and eval_dataset is not None:
        metrics.update(trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix=args.eval_split))
    if trainer.is_world_process_zero():
        atomic_json(args.output_dir / "metrics.json", metrics)
        resolved = json.loads((args.output_dir / "resolved_config.json").read_text())
        expected_final = args.stop_after_step is None and trainer.state.global_step == args.max_steps
        atomic_json(args.output_dir / "run_manifest.json", {
            **resolved,
            "status": "complete" if expected_final else "cleanly_stopped",
            "global_step": trainer.state.global_step,
            "peak_reserved_gb": metrics["peak_reserved_gb"],
            "ref_model_is_none": trainer.ref_model is None,
        })
        trainer.save_state()


if __name__ == "__main__":
    main()
