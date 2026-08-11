from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from trl.trainer.dpo_trainer import DPOTrainer, DataCollatorForPreference
from trl.trainer.utils import selective_log_softmax
from trl.trainer.utils import flush_left

from .constants import MODEL_REVISION, QWEN_PAD_ID, TRL_COMMIT
from .utils import atomic_json, canonical_json, ordered_id_root, sha256_file, sha256_text


def _init_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def _score_pair_batch(model, rows: list[dict[str, Any]], device: torch.device) -> tuple[list[float], list[float], list[int], list[int]]:
    """Mirror TRL 0.19.1 concatenated_forward exactly, without a Trainer/ref model."""
    collator = DataCollatorForPreference(pad_token_id=QWEN_PAD_ID)
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in collator(rows).items()}
    concatenated = DPOTrainer.concatenated_inputs(batch, padding_value=QWEN_PAD_ID)
    prompt_ids = concatenated["prompt_input_ids"]
    prompt_mask = concatenated["prompt_attention_mask"]
    completion_ids = concatenated["completion_input_ids"]
    completion_mask = concatenated["completion_attention_mask"]
    input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
    attention_mask = torch.cat((prompt_mask, completion_mask), dim=1)
    loss_mask = torch.cat((torch.zeros_like(prompt_mask), completion_mask), dim=1)
    attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
    first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
    logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1
    # Full-parameter TRL 0.19.1 calls concatenated_forward without autocast;
    # keeping this context identical also preserves its BF16 reduction behavior.
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask,
            logits_to_keep=logits_to_keep, use_cache=False,
        ).logits
        labels = torch.roll(input_ids, shifts=-1, dims=1)[:, -logits_to_keep:]
        aligned_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()[:, -logits_to_keep:]
        labels[~aligned_mask] = 0
        token_logps = selective_log_softmax(logits, labels)
        token_logps[~aligned_mask] = 0
        token_logps = torch.roll(token_logps, shifts=1, dims=1)
        sums = token_logps[:, 1:].sum(-1).float().cpu()
    count = len(rows)
    return (
        sums[:count].tolist(), sums[count:].tolist(),
        [len(row["chosen_input_ids"]) for row in rows],
        [len(row["rejected_input_ids"]) for row in rows],
    )


def build_reference_cache(
    dataset_dir: Path,
    model_path: Path,
    output_dir: Path,
    tokenization_manifest: Path,
    batch_size: int = 2,
) -> dict[str, Any] | None:
    rank, world, local_rank = _init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    existing_payload: list[dict[str, Any] | None] = [None]
    if rank == 0 and (output_dir / "manifest.json").exists():
        candidate = json.loads((output_dir / "manifest.json").read_text())
        if candidate.get("status") == "complete":
            existing_payload[0] = candidate
    if world > 1:
        dist.broadcast_object_list(existing_payload, src=0)
    if existing_payload[0] is not None:
        if world > 1:
            dist.destroy_process_group()
        return existing_payload[0] if rank == 0 else None
    if rank == 0:
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite incomplete reference cache {output_dir}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        build_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
        build_dir.mkdir(parents=True, exist_ok=True)
    else:
        build_dir = Path(".")
    if world > 1:
        payload = [str(build_dir)]
        dist.broadcast_object_list(payload, src=0)
        build_dir = Path(payload[0])
    parts_dir = build_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", use_cache=False,
    ).to(device).eval()
    assigned: list[dict[str, Any]] = []
    global_index = 0
    for split in ("train", "validation", "test"):
        for row in pq.read_table(dataset_dir / f"{split}.parquet").to_pylist():
            if global_index % world == rank:
                row["split"] = split
                assigned.append(row)
            global_index += 1
    result: list[dict[str, Any]] = []
    for start in range(0, len(assigned), batch_size):
        batch = assigned[start:start + batch_size]
        chosen, rejected, chosen_counts, rejected_counts = _score_pair_batch(model, batch, device)
        for row, c, r, nc, nr in zip(batch, chosen, rejected, chosen_counts, rejected_counts):
            if not torch.isfinite(torch.tensor([c, r])).all():
                raise FloatingPointError(f"non-finite reference logp for {row['pair_id']}")
            result.append({
                "pair_id": row["pair_id"], "split": row["split"],
                "ref_chosen_logps": float(c), "ref_rejected_logps": float(r),
                "chosen_token_count": nc, "rejected_token_count": nr,
            })
    pq.write_table(pa.Table.from_pylist(result), parts_dir / f"part-{rank:05d}.parquet", compression="zstd")
    if world > 1:
        dist.barrier()
    manifest = None
    if rank == 0:
        rows = []
        for path in sorted(parts_dir.glob("part-*.parquet")):
            rows.extend(pq.read_table(path).to_pylist())
        expected = sum(pq.read_metadata(dataset_dir / f"{split}.parquet").num_rows for split in ("train", "validation", "test"))
        if len(rows) != expected or len({row["pair_id"] for row in rows}) != expected:
            raise AssertionError(f"reference cache coverage/uniqueness failed: rows={len(rows)}, expected={expected}")
        rows.sort(key=lambda row: (row["split"], row["pair_id"]))
        cache_path = build_dir / "reference_logps.parquet"
        pq.write_table(pa.Table.from_pylist(rows), cache_path, compression="zstd", compression_level=6)
        token_manifest = json.loads(tokenization_manifest.read_text())
        chosen_values = np.asarray([row["ref_chosen_logps"] for row in rows], dtype=np.float64)
        rejected_values = np.asarray([row["ref_rejected_logps"] for row in rows], dtype=np.float64)
        manifest = {
            "status": "complete", "model_revision": MODEL_REVISION, "trl_commit": TRL_COMMIT,
            "dtype": "bfloat16_forward_bfloat16_sum_float32_storage", "completion_logp_reduction": "token_sum",
            "rows": len(rows), "unique_pair_ids": len(rows),
            "split_rows": {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")},
            "finite": bool(np.isfinite(chosen_values).all() and np.isfinite(rejected_values).all()),
            "statistics": {
                "chosen_logp_min": float(chosen_values.min()), "chosen_logp_max": float(chosen_values.max()), "chosen_logp_mean": float(chosen_values.mean()),
                "rejected_logp_min": float(rejected_values.min()), "rejected_logp_max": float(rejected_values.max()), "rejected_logp_mean": float(rejected_values.mean()),
            },
            "ordered_pair_id_root": ordered_id_root([row["pair_id"] for row in rows]),
            "tokenization_root_digest": token_manifest["root_digest"],
            "tokenizer_chat_template_sha256": token_manifest["tokenizer_chat_template_sha256"],
            "enable_thinking": False, "max_length": token_manifest["max_length"],
            "cache_sha256": sha256_file(cache_path),
        }
        manifest["root_digest"] = sha256_text(canonical_json(manifest))
        atomic_json(build_dir / "manifest.json", manifest)
        shutil.rmtree(parts_dir)
        os.replace(build_dir, output_dir)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return manifest


def join_reference_cache(dataset_rows: list[dict[str, Any]], cache_path: Path, split: str) -> list[dict[str, Any]]:
    cached = {
        row["pair_id"]: row for row in pq.read_table(cache_path).to_pylist() if row["split"] == split
    }
    ids = [row["pair_id"] for row in dataset_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset pair IDs are not unique")
    if set(ids) != set(cached):
        missing = set(ids) - set(cached)
        extra = set(cached) - set(ids)
        raise ValueError(f"reference cache 1:1 join failed: missing={len(missing)} extra={len(extra)}")
    joined = []
    for row in dataset_rows:
        ref = cached[row["pair_id"]]
        joined.append({**row, "ref_chosen_logps": ref["ref_chosen_logps"], "ref_rejected_logps": ref["ref_rejected_logps"]})
    return joined
