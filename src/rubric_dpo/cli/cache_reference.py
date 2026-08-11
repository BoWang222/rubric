from __future__ import annotations

import argparse
import json
from pathlib import Path

from rubric_dpo.reference import build_reference_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Build durable pair-keyed Qwen3 reference log-prob cache")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    manifest = build_reference_cache(
        args.dataset_dir, args.model, args.output,
        args.dataset_dir / "tokenization.json", args.batch_size,
    )
    if manifest is not None:
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
