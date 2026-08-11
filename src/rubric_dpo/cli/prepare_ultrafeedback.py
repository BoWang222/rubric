from __future__ import annotations

import argparse
import json
from pathlib import Path

from rubric_dpo.data.tokenization import tokenize_dataset
from rubric_dpo.data.ultrafeedback import prepare_ultrafeedback


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audited UltraFeedback v2 and materialized Qwen3 tokens")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--h4-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    base = args.output / "base"
    tokens = args.output / "tokens_qwen3_8b_non_thinking"
    if (base / "manifest.json").exists() and not args.overwrite:
        base_manifest = json.loads((base / "manifest.json").read_text())
        if base_manifest.get("status") != "complete":
            raise RuntimeError("existing base dataset is not complete")
    else:
        base_manifest = prepare_ultrafeedback(args.raw_dir, args.h4_dir, base, args.overwrite)
    if (tokens / "tokenization.json").exists() and not args.overwrite:
        token_manifest = json.loads((tokens / "tokenization.json").read_text())
        if token_manifest.get("status") != "complete":
            raise RuntimeError("existing tokenized dataset is not complete")
    else:
        token_manifest = tokenize_dataset(base, args.model, tokens, args.overwrite)
    print(json.dumps({"base": base_manifest, "tokens": token_manifest}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
