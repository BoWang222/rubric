"""DEPRECATED archival pipeline.

Do not use for production: this file consumes H4 overall-score gaps and a fixed
scale, which violates the audited UltraFeedback v2 four-aspect contract. Use
``rubric-prepare-ultrafeedback`` instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_dataset_builder


DEFAULT_DATASET_ID = "HuggingFaceH4/ultrafeedback_binarized"
DEFAULT_REVISION = "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85"
DEFAULT_OUTPUT = Path(
    "/home/chunyuan/rubric/data/processed/ultrafeedback_binarized_v1"
)
DEFAULT_VALIDATION_PROMPTS = 2_000
DEFAULT_SEED = 42
MARGIN_SCALE = 9.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the UltraFeedback positive-margin DPO data contract."
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validation-prompts",
        type=int,
        default=DEFAULT_VALIDATION_PROMPTS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def stable_prompt_rank(prompt_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{prompt_id}".encode("utf-8")).hexdigest()


def stable_pair_id(row: dict[str, Any]) -> str:
    payload = {
        "prompt_id": row["prompt_id"],
        "chosen": row["chosen"],
        "rejected": row["rejected"],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def add_contract_columns(
    row: dict[str, Any],
    split: str,
    dataset_id: str,
) -> dict[str, Any]:
    margin_raw = float(row["score_chosen"] - row["score_rejected"])
    return {
        "pair_id": stable_pair_id(row),
        "dataset_id": dataset_id,
        "rubric_system_id": "ultrafeedback_binarized_overall",
        "margin_raw": margin_raw,
        "margin_normalized": margin_raw / MARGIN_SCALE,
        "split": split,
    }


def enrich(dataset: Dataset, split: str, dataset_id: str) -> Dataset:
    return dataset.map(
        lambda row: add_contract_columns(row, split, dataset_id),
        desc=f"Adding frozen contract columns: {split}",
    )


def prompt_ids(dataset: Dataset) -> set[str]:
    return set(dataset["prompt_id"])


def assert_disjoint_prompt_ids(
    train: Dataset,
    validation: Dataset,
    test: Dataset,
) -> dict[str, int]:
    train_ids = prompt_ids(train)
    validation_ids = prompt_ids(validation)
    test_ids = prompt_ids(test)

    overlaps = {
        "train_validation": len(train_ids & validation_ids),
        "train_test": len(train_ids & test_ids),
        "validation_test": len(validation_ids & test_ids),
    }
    if any(overlaps.values()):
        raise ValueError(f"Prompt leakage detected: {overlaps}")
    return overlaps


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def hash_saved_files(output: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    root_hasher = hashlib.sha256()

    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "content_hashes.json":
            continue

        relative = path.relative_to(output).as_posix()
        file_hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_hasher.update(chunk)

        digest = file_hasher.hexdigest()
        size = path.stat().st_size
        files.append({"path": relative, "bytes": size, "sha256": digest})
        root_hasher.update(f"{relative}\t{size}\t{digest}\n".encode("utf-8"))

    return {
        "algorithm": "sha256",
        "root_digest": root_hasher.hexdigest(),
        "files": files,
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(
            f"{output} already exists; do not overwrite a frozen dataset contract"
        )

    builder = load_dataset_builder(args.dataset_id, revision=args.revision)
    train_source = load_dataset(
        args.dataset_id,
        revision=args.revision,
        split="train_prefs",
    )
    test_source = load_dataset(
        args.dataset_id,
        revision=args.revision,
        split="test_prefs",
    )

    train_positive = train_source.filter(
        lambda row: row["score_chosen"] > row["score_rejected"],
        desc="Filtering positive-margin train pairs",
    )
    train_ties = train_source.filter(
        lambda row: row["score_chosen"] == row["score_rejected"],
        desc="Collecting train ties",
    )
    train_negative = len(train_source) - len(train_positive) - len(train_ties)

    test_positive = test_source.filter(
        lambda row: row["score_chosen"] > row["score_rejected"],
        desc="Filtering positive-margin test pairs",
    )
    test_ties = test_source.filter(
        lambda row: row["score_chosen"] == row["score_rejected"],
        desc="Collecting test ties",
    )
    test_negative = len(test_source) - len(test_positive) - len(test_ties)

    if train_negative or test_negative:
        raise ValueError(
            "The binarized source contains negative-margin rows: "
            f"train={train_negative}, test={test_negative}"
        )

    unique_train_prompt_ids = sorted(set(train_positive["prompt_id"]))
    if args.validation_prompts <= 0:
        raise ValueError("--validation-prompts must be positive")
    if args.validation_prompts >= len(unique_train_prompt_ids):
        raise ValueError("Validation prompt count leaves no training prompts")

    ranked_prompt_ids = sorted(
        unique_train_prompt_ids,
        key=lambda prompt_id: stable_prompt_rank(prompt_id, args.seed),
    )
    validation_prompt_ids = set(ranked_prompt_ids[: args.validation_prompts])

    validation = train_positive.filter(
        lambda row: row["prompt_id"] in validation_prompt_ids,
        desc="Selecting deterministic validation prompts",
    )
    train = train_positive.filter(
        lambda row: row["prompt_id"] not in validation_prompt_ids,
        desc="Selecting deterministic training prompts",
    )

    train = enrich(train, "train", args.dataset_id)
    validation = enrich(validation, "validation", args.dataset_id)
    test = enrich(test_positive, "test", args.dataset_id)
    train_ties = enrich(train_ties, "train_ties", args.dataset_id)
    test_ties = enrich(test_ties, "test_ties", args.dataset_id)

    overlaps = assert_disjoint_prompt_ids(train, validation, test)

    frozen = DatasetDict(
        {
            "train": train,
            "validation": validation,
            "test": test,
            "train_ties": train_ties,
            "test_ties": test_ties,
        }
    )
    frozen.save_to_disk(str(output))

    contract = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_repository": args.dataset_id,
        "dataset_revision": args.revision,
        "dataset_config": builder.config.name,
        "license": builder.info.license,
        "source_splits": {
            "train_prefs": len(train_source),
            "test_prefs": len(test_source),
        },
        "filter_contract": {
            "main_splits": "score_chosen > score_rejected",
            "ties": "score_chosen == score_rejected, saved separately",
            "negative_margin_rows": {
                "train": train_negative,
                "test": test_negative,
            },
        },
        "split_contract": {
            "train_and_validation_source": "train_prefs positive-margin rows",
            "validation_rule": (
                "first N unique prompt_id values after sorting by "
                "sha256(f'{seed}:{prompt_id}')"
            ),
            "validation_prompt_count_requested": args.validation_prompts,
            "seed": args.seed,
            "test_source": "official test_prefs positive-margin rows",
            "prompt_overlap_counts": overlaps,
        },
        "margin_contract": {
            "margin_raw": "score_chosen - score_rejected",
            "margin_normalized": "margin_raw / 9.0",
            "scale": MARGIN_SCALE,
        },
        "rubric_contract": {
            "rubric_system_id": "ultrafeedback_binarized_overall",
            "scope": "overall scalar score only",
            "criterion_level_scores_available": False,
            "note": (
                "Nested 1D/2D/3D/4D ROIV experiments require a later "
                "alignment with openbmb/UltraFeedback raw aspect scores."
            ),
        },
        "processed_rows": {name: len(dataset) for name, dataset in frozen.items()},
        "processed_unique_prompts": {
            name: len(prompt_ids(dataset)) for name, dataset in frozen.items()
        },
        "fingerprints": {
            name: dataset._fingerprint for name, dataset in frozen.items()
        },
        "columns": {name: dataset.column_names for name, dataset in frozen.items()},
        "output_path": str(output),
    }
    write_json(output / "data_contract.json", contract)

    hashes = hash_saved_files(output)
    write_json(output / "content_hashes.json", hashes)

    summary = {
        "status": "DATA_CONTRACT_COMPLETE",
        "output": str(output),
        "rows": contract["processed_rows"],
        "unique_prompts": contract["processed_unique_prompts"],
        "prompt_overlap_counts": overlaps,
        "root_digest": hashes["root_digest"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
