from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from ..constants import (
    MAX_COMPLETION_LENGTH,
    MAX_LENGTH,
    MAX_PROMPT_LENGTH,
    MODEL_REVISION,
    QWEN_EOS_ID,
    QWEN_PAD_ID,
)
from ..utils import atomic_json, canonical_json, ordered_id_root, sha256_file, sha256_text

CONTROL_STRINGS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")


def _completion_suffix(tokenizer, prompt_messages, response: str) -> tuple[list[int], list[int]]:
    if any(control in response for control in CONTROL_STRINGS):
        raise ValueError("assistant response contains a Qwen control string")
    prompt_ids = list(tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    ))
    full_ids = list(tokenizer.apply_chat_template(
        prompt_messages + [{"role": "assistant", "content": response}],
        tokenize=True, add_generation_prompt=False, enable_thinking=False,
    ))
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise AssertionError("full Qwen3 conversation is not prefixed by the rendered prompt")
    completion = full_ids[len(prompt_ids):]
    if QWEN_EOS_ID in completion:
        eos_index = completion.index(QWEN_EOS_ID)
        tail = completion[eos_index + 1:]
        if tail and tokenizer.decode(tail).strip():
            raise AssertionError("non-whitespace tokens follow the assistant EOS")
        completion = completion[: eos_index + 1]
    else:
        completion.append(QWEN_EOS_ID)
    if completion.count(QWEN_EOS_ID) != 1 or completion[-1] != QWEN_EOS_ID:
        raise AssertionError("completion must contain exactly one terminal EOS")
    return prompt_ids, completion


def _truncate(prompt: list[int], completion: list[int]) -> tuple[list[int], list[int], bool, bool]:
    prompt_truncated = len(prompt) > MAX_PROMPT_LENGTH
    prompt = prompt[-MAX_PROMPT_LENGTH:]
    budget = min(MAX_COMPLETION_LENGTH, MAX_LENGTH - len(prompt))
    if budget < 1:
        raise AssertionError("no token budget remains for completion EOS")
    completion_truncated = len(completion) > budget
    if completion_truncated:
        completion = completion[: budget - 1] + [QWEN_EOS_ID]
    return prompt, completion, prompt_truncated, completion_truncated


def tokenize_dataset(base_dir: Path, model_path: Path, output_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id != QWEN_PAD_ID or tokenizer.eos_token_id != QWEN_EOS_ID:
        raise AssertionError(
            f"Qwen token ids changed: pad={tokenizer.pad_token_id}, eos={tokenizer.eos_token_id}"
        )
    tokenizer.padding_side = "right"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    manifests: dict[str, Any] = {}
    try:
        for split in ("train", "validation", "test"):
            rows = pq.read_table(base_dir / f"{split}.parquet").to_pylist()
            tokenized: list[dict[str, Any]] = []
            stats: Counter = Counter()
            prompt_lengths: list[int] = []
            chosen_lengths: list[int] = []
            rejected_lengths: list[int] = []
            for row in rows:
                prompt_messages = row["prompt_messages"]
                prompt_c, chosen = _completion_suffix(tokenizer, prompt_messages, row["chosen_response"])
                prompt_r, rejected = _completion_suffix(tokenizer, prompt_messages, row["rejected_response"])
                if prompt_c != prompt_r:
                    raise AssertionError("chosen/rejected prompt rendering differs")
                prompt_lengths.append(len(prompt_c)); chosen_lengths.append(len(chosen)); rejected_lengths.append(len(rejected))
                prompt, chosen, prompt_tc, chosen_tc = _truncate(prompt_c, chosen)
                prompt_again, rejected, prompt_tr, rejected_tc = _truncate(prompt_r, rejected)
                if prompt != prompt_again:
                    raise AssertionError("chosen/rejected prompt truncation differs")
                stats["prompt_truncated"] += int(prompt_tc or prompt_tr)
                stats["chosen_truncated"] += int(chosen_tc)
                stats["rejected_truncated"] += int(rejected_tc)
                if len(chosen) < 1 or len(rejected) < 1:
                    raise AssertionError("empty completion after tokenization")
                tokenized.append({
                    "pair_id": row["pair_id"],
                    "split": split,
                    "prompt_input_ids": prompt,
                    "chosen_input_ids": chosen,
                    "rejected_input_ids": rejected,
                    "margin_raw": row["margin_raw"],
                    "margin_normalized": row["margin_normalized"],
                    "sample_weight": row["sample_weight"],
                    "q95_train": row["q95_train"],
                    "mu_train": row["mu_train"],
                    "chosen_token_count": len(chosen),
                    "rejected_token_count": len(rejected),
                    "chosen_character_count": len(row["chosen_response"]),
                    "rejected_character_count": len(row["rejected_response"]),
                })
            path = temporary / f"{split}.parquet"
            pq.write_table(pa.Table.from_pylist(tokenized), path, compression="zstd", compression_level=6)
            def percentiles(values: list[int]) -> dict[str, float]:
                import numpy as np
                return {name: float(np.quantile(values, q, method="linear")) for name, q in (("p50", .5), ("p95", .95), ("p99", .99), ("max", 1.0))}
            manifests[split] = {
                "rows": len(tokenized), "ordered_pair_id_root": ordered_id_root([row["pair_id"] for row in tokenized]),
                "truncation": dict(stats), "prompt_lengths": percentiles(prompt_lengths),
                "chosen_lengths": percentiles(chosen_lengths), "rejected_lengths": percentiles(rejected_lengths),
            }
        smoke_ids = set((base_dir / "smoke_1024_pair_ids.txt").read_text().splitlines())
        train_rows = pq.read_table(temporary / "train.parquet").to_pylist()
        smoke_order = (base_dir / "smoke_1024_pair_ids.txt").read_text().splitlines()
        by_id = {row["pair_id"]: row for row in train_rows if row["pair_id"] in smoke_ids}
        if set(by_id) != smoke_ids:
            raise AssertionError("tokenized smoke subset coverage mismatch")
        pq.write_table(pa.Table.from_pylist([by_id[pair_id] for pair_id in smoke_order]), temporary / "smoke_1024.parquet", compression="zstd")
        template_hash = sha256_text(str(tokenizer.chat_template))
        files = {path.name: sha256_file(path) for path in sorted(temporary.glob("*.parquet"))}
        manifest = {
            "status": "complete", "model_revision": MODEL_REVISION,
            "tokenizer_class": tokenizer.__class__.__name__, "tokenizer_chat_template_sha256": template_hash,
            "enable_thinking": False, "pad_token_id": QWEN_PAD_ID, "eos_token_id": QWEN_EOS_ID,
            "max_prompt_length": MAX_PROMPT_LENGTH, "max_completion_length": MAX_COMPLETION_LENGTH,
            "max_length": MAX_LENGTH, "prompt_truncation": "keep_end", "completion_truncation": "keep_start_keep_eos",
            "completion_logp_reduction": "token_sum", "splits": manifests, "files": files,
        }
        manifest["root_digest"] = sha256_text(canonical_json(manifest))
        atomic_json(temporary / "tokenization.json", manifest)
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(output_dir)
            backup = output_dir.with_name(output_dir.name + ".previous")
            if backup.exists():
                raise FileExistsError(f"backup already exists: {backup}")
            os.replace(output_dir, backup)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return json.loads((output_dir / "tokenization.json").read_text())
