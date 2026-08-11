from __future__ import annotations

import glob
import itertools
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..constants import ASPECTS, EXPECTED, H4_REVISION, RAW_REVISION
from ..utils import atomic_json, canonical_json, ordered_id_root, sha256_file, sha256_text


def _assistant_text(messages: list[dict[str, Any]], prompt: str) -> str:
    if len(messages) != 2 or messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        raise ValueError("H4 chosen/rejected must contain exactly user then assistant")
    if messages[0].get("content") != prompt:
        raise ValueError("H4 message prompt does not match prompt column")
    return str(messages[1]["content"])


def _ratings(completion: dict[str, Any]) -> tuple[Any, ...]:
    annotations = completion.get("annotations") or {}
    return tuple((annotations.get(name) or {}).get("Rating") for name in ASPECTS)


def _parse_ratings(values: tuple[Any, ...]) -> tuple[float, ...] | None:
    parsed = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or not 1.0 <= number <= 5.0:
            return None
        parsed.append(number)
    return tuple(parsed)


def _raw_index(raw_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    files = sorted(glob.glob(str(raw_dir / "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No UltraFeedback JSONL files under {raw_dir}")
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows = completions = zero_completion_rows = 0
    for file_name in files:
        with Path(file_name).open(encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                raw = json.loads(line)
                instruction = str(raw["instruction"])
                cs = raw.get("completions") or []
                index[sha256_text(instruction)].append(
                    {
                        "file": Path(file_name).name,
                        "line": line_index,
                        "instruction": instruction,
                        "completions": cs,
                    }
                )
                rows += 1
                completions += len(cs)
                zero_completion_rows += int(not cs)
    if rows != 63967 or completions != 255864 or zero_completion_rows != 1:
        raise AssertionError(
            f"raw contract mismatch: rows={rows}, completions={completions}, zero={zero_completion_rows}"
        )
    return index, {
        "files": [Path(name).name for name in files],
        "rows": rows,
        "completions": completions,
        "zero_completion_rows": zero_completion_rows,
    }


def _matches(completions: list[dict[str, Any]], response: str, overall: float) -> list[dict[str, Any]]:
    result = []
    for completion in completions:
        if completion.get("response") != response:
            continue
        try:
            score = float(completion.get("overall_score"))
        except (TypeError, ValueError):
            continue
        if math.isclose(score, overall, rel_tol=0.0, abs_tol=1e-12):
            result.append(completion)
    return result


def _pair_id(row: dict[str, Any]) -> str:
    identity = {
        "h4_revision": H4_REVISION,
        "source_split": row["source_split"],
        "source_row_index": row["source_row_index"],
        "prompt_id": row["prompt_id"],
        "chosen_sha256": sha256_text(row["chosen_response"]),
        "rejected_sha256": sha256_text(row["rejected_response"]),
    }
    return sha256_text(canonical_json(identity))


def _align_split(path: Path, source_split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    raw_index = _align_split.raw_index  # type: ignore[attr-defined]
    source_rows = pq.read_table(path).to_pylist()
    accepted: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    counts: Counter = Counter()
    for source_row_index, source in enumerate(source_rows):
        prompt = str(source["prompt"])
        prompt_id = str(source["prompt_id"])
        if prompt_id != sha256_text(prompt):
            raise AssertionError(f"prompt_id mismatch at {source_split}:{source_row_index}")
        chosen_response = _assistant_text(source["chosen"], prompt)
        rejected_response = _assistant_text(source["rejected"], prompt)
        feasible: list[tuple[tuple[Any, ...], tuple[Any, ...], str, int]] = []
        for raw in raw_index.get(prompt_id, []):
            chosen = _matches(raw["completions"], chosen_response, float(source["score_chosen"]))
            rejected = _matches(raw["completions"], rejected_response, float(source["score_rejected"]))
            for chosen_item, rejected_item in itertools.product(chosen, rejected):
                feasible.append((_ratings(chosen_item), _ratings(rejected_item), raw["file"], raw["line"]))
        tuple_set = {(canonical_json(x[0]), canonical_json(x[1])) for x in feasible}
        base = {
            "source_split": source_split,
            "source_row_index": source_row_index,
            "prompt_id": prompt_id,
            "prompt": prompt,
            "chosen_response": chosen_response,
            "rejected_response": rejected_response,
            "chosen_overall_audit": float(source["score_chosen"]),
            "rejected_overall_audit": float(source["score_rejected"]),
        }
        if not feasible:
            counts["no_feasible_alignment"] += 1
            quarantine.append({**base, "reason": "no_feasible_alignment"})
            continue
        # One duplicated source row has many locator combinations but no numeric
        # rating vector on either branch. There is no semantic tuple that could
        # be disambiguated, so the frozen audit classifies it as invalid rather
        # than ambiguous. If either branch has a valid candidate, ordinary
        # tuple-set ambiguity retains precedence below.
        if (
            not any(_parse_ratings(item[0]) is not None for item in feasible)
            and not any(_parse_ratings(item[1]) is not None for item in feasible)
        ):
            counts["invalid_rating"] += 1
            quarantine.append({**base, "reason": "invalid_rating", "alignment_candidate_count": len(feasible)})
            continue
        if len(tuple_set) != 1:
            counts["ambiguous_alignment"] += 1
            quarantine.append({**base, "reason": "ambiguous_alignment", "alignment_candidate_count": len(feasible)})
            continue
        chosen_raw, rejected_raw = feasible[0][0], feasible[0][1]
        chosen_scores = _parse_ratings(chosen_raw)
        rejected_scores = _parse_ratings(rejected_raw)
        if chosen_scores is None or rejected_scores is None:
            counts["invalid_rating"] += 1
            quarantine.append({**base, "reason": "invalid_rating", "alignment_candidate_count": len(feasible)})
            continue
        chosen_mean = float(np.mean(chosen_scores))
        rejected_mean = float(np.mean(rejected_scores))
        margin_raw = chosen_mean - rejected_mean
        if margin_raw < 0:
            counts["direction_conflict"] += 1
            quarantine.append({**base, "reason": "direction_conflict", "margin_raw": margin_raw})
            continue
        if margin_raw == 0:
            counts["exact_tie"] += 1
            quarantine.append({**base, "reason": "exact_tie", "margin_raw": margin_raw})
            continue
        counts["positive"] += 1
        row = {
            **base,
            "dataset_id": "HuggingFaceH4/ultrafeedback_binarized",
            "dataset_revision": H4_REVISION,
            "raw_revision": RAW_REVISION,
            "prompt_messages": [{"role": "user", "content": prompt}],
            "chosen_messages": [{"role": "assistant", "content": chosen_response}],
            "rejected_messages": [{"role": "assistant", "content": rejected_response}],
            "chosen_aspect_scores": list(chosen_scores),
            "rejected_aspect_scores": list(rejected_scores),
            "chosen_aspect_mean": chosen_mean,
            "rejected_aspect_mean": rejected_mean,
            "margin_raw": margin_raw,
            "rubric_system_id": "ultrafeedback_native_four_aspect_equal_v1",
            "rubric_id": sha256_text(canonical_json({"aspects": ASPECTS, "aggregation": "equal_mean"})),
            "alignment_candidate_count": len(feasible),
        }
        row["pair_id"] = _pair_id(row)
        accepted.append(row)
    counts["source_rows"] = len(source_rows)
    return accepted, quarantine, counts


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd", compression_level=6)


def prepare_ultrafeedback(raw_dir: Path, h4_dir: Path, output_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; pass --overwrite only for an unclaimed build")
    raw_index, raw_stats = _raw_index(raw_dir)
    _align_split.raw_index = raw_index  # type: ignore[attr-defined]
    train_all, train_quarantine, train_counts = _align_split(
        h4_dir / "data/train_prefs-00000-of-00001.parquet", "train_prefs"
    )
    test_rows, test_quarantine, test_counts = _align_split(
        h4_dir / "data/test_prefs-00000-of-00001.parquet", "test_prefs"
    )
    expected_train = {
        "source_rows": EXPECTED["train_source_rows"],
        "positive": EXPECTED["train_positive"],
        "exact_tie": EXPECTED["train_exact_tie"],
        "direction_conflict": EXPECTED["train_direction_conflict"],
        "invalid_rating": EXPECTED["train_invalid_rating"],
        "ambiguous_alignment": EXPECTED["train_ambiguous_alignment"],
        "no_feasible_alignment": 0,
    }
    for key, expected in expected_train.items():
        if train_counts[key] != expected:
            raise AssertionError(f"train {key}: expected {expected}, got {train_counts[key]}")
    if test_counts["source_rows"] != EXPECTED["test_source_rows"] or test_counts["positive"] != EXPECTED["test"]:
        raise AssertionError(f"test count mismatch: {dict(test_counts)}")

    deduped: dict[str, dict[str, Any]] = {}
    for row in sorted(train_all, key=lambda value: value["source_row_index"]):
        deduped.setdefault(row["prompt_id"], row)
    dedup_drop = len(train_all) - len(deduped)
    if dedup_drop != EXPECTED["train_positive_dedup_drop"]:
        raise AssertionError(f"expected 5 positive prompt duplicates, got {dedup_drop}")
    ordered = sorted(deduped.values(), key=lambda row: sha256_text(f"42:{row['prompt_id']}"))
    validation_rows, train_rows = ordered[:2000], ordered[2000:]
    if (len(train_rows), len(validation_rows), len(test_rows)) != (
        EXPECTED["train"], EXPECTED["validation"], EXPECTED["test"]
    ):
        raise AssertionError("final split counts do not match the frozen contract")
    prompt_sets = [set(row["prompt_id"] for row in rows) for rows in (train_rows, validation_rows, test_rows)]
    if prompt_sets[0] & prompt_sets[1] or prompt_sets[0] & prompt_sets[2] or prompt_sets[1] & prompt_sets[2]:
        raise AssertionError("train/validation/test prompt leakage")

    q95 = float(np.quantile([row["margin_raw"] for row in train_rows], 0.95, method="linear"))
    if q95 != 3.5:
        raise AssertionError(f"expected train-only q95=3.5, got {q95}")
    for split, rows in (("train", train_rows), ("validation", validation_rows), ("test", test_rows)):
        for row in rows:
            row["split"] = split
            row["q95_train"] = q95
            row["margin_normalized"] = float(np.clip(row["margin_raw"] / q95, 0.0, 1.0))
        near_ties = sum(row["margin_normalized"] < 0.05 for row in rows)
        if near_ties:
            raise AssertionError(f"unexpected normalized near ties in {split}: {near_ties}")
    mu_train = float(np.mean([row["margin_normalized"] for row in train_rows]))
    for rows in (train_rows, validation_rows, test_rows):
        for row in rows:
            row["mu_train"] = mu_train
            row["sample_weight"] = row["margin_normalized"] / mu_train
    if not math.isclose(np.mean([row["sample_weight"] for row in train_rows]), 1.0, abs_tol=1e-12):
        raise AssertionError("train sample weights do not have mean one")
    for split, rows in (("train", train_rows), ("validation", validation_rows), ("test", test_rows)):
        expected = EXPECTED[f"{split}_normalized_one"]
        actual = sum(row["margin_normalized"] == 1.0 for row in rows)
        if actual != expected:
            raise AssertionError(f"{split} normalized-one count expected {expected}, got {actual}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for split, rows in (("train", train_rows), ("validation", validation_rows), ("test", test_rows)):
            _write_parquet(temporary / f"{split}.parquet", rows)
        _write_parquet(temporary / "quarantine.parquet", train_quarantine + test_quarantine)
        smoke = sorted(train_rows, key=lambda row: sha256_text(f"smoke:42:{row['pair_id']}"))[:1024]
        smoke_ids = [row["pair_id"] for row in smoke]
        (temporary / "smoke_1024_pair_ids.txt").write_text("\n".join(smoke_ids) + "\n", encoding="utf-8")
        split_manifest = {
            "train": {"rows": len(train_rows), "ordered_pair_id_root": ordered_id_root([r["pair_id"] for r in train_rows])},
            "validation": {"rows": len(validation_rows), "ordered_pair_id_root": ordered_id_root([r["pair_id"] for r in validation_rows])},
            "test": {"rows": len(test_rows), "ordered_pair_id_root": ordered_id_root([r["pair_id"] for r in test_rows])},
            "smoke_1024": {"rows": 1024, "ordered_pair_id_root": ordered_id_root(smoke_ids)},
        }
        atomic_json(temporary / "sources.json", {
            "raw": {"id": "openbmb/UltraFeedback", "revision": RAW_REVISION, **raw_stats},
            "pairs": {"id": "HuggingFaceH4/ultrafeedback_binarized", "revision": H4_REVISION},
        })
        atomic_json(temporary / "alignment.json", {
            "policy": "prompt_sha + exact_response + overall_abs_tol_1e-12; overall audit only",
            "filter_priority": "no feasible; both branches have no numeric candidate; ambiguous tuple; invalid unique tuple; nonpositive gap; prompt dedup",
            "aspect_order": list(ASPECTS),
            "train": dict(train_counts), "test": dict(test_counts), "positive_prompt_dedup_drop": dedup_drop,
        })
        atomic_json(temporary / "splits.json", split_manifest)
        atomic_json(temporary / "margins.json", {
            "aggregation": "equal arithmetic mean of four native aspects",
            "q95_train": q95, "quantile_method": "numpy.linear", "mu_train": mu_train,
            "normalization": "clip(margin_raw/q95_train,0,1)", "near_tie_threshold": 0.05,
            "sample_weight": "margin_normalized/mu_train; no batch renormalization",
        })
        files = sorted(temporary.glob("*.parquet"))
        content_hashes = {path.name: sha256_file(path) for path in files}
        content_hashes["smoke_1024_pair_ids.txt"] = sha256_file(temporary / "smoke_1024_pair_ids.txt")
        root_digest = sha256_text(canonical_json(content_hashes))
        atomic_json(temporary / "content_hashes.json", {"files": content_hashes, "root_digest": root_digest})
        atomic_json(temporary / "manifest.json", {
            "status": "complete", "contract": "ultrafeedback_v2", "root_digest": root_digest,
            "q95_train": q95, "mu_train": mu_train, "splits": split_manifest,
        })
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(output_dir)
            backup = output_dir.with_name(output_dir.name + ".previous")
            if backup.exists():
                raise FileExistsError(f"Refusing overwrite because backup already exists: {backup}")
            os.replace(output_dir, backup)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
