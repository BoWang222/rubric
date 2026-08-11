from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from rubric_dpo.utils import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate verified local baseline diagnostics")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for manifest_path in sorted(args.runs_root.glob("**/run_manifest.json")):
        run_dir = manifest_path.parent
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        runs.append({"run_dir": str(run_dir), "variant": manifest["variant"], "seed": manifest["seed"], **metrics})
    if not runs:
        raise FileNotFoundError("no completed run metrics found")
    summary = {"status": "local diagnostic", "runs": runs, "aggregate": {}}
    for variant in sorted({row["variant"] for row in runs}):
        subset = [row for row in runs if row["variant"] == variant]
        numeric_keys = set.intersection(*[{key for key, value in row.items() if isinstance(value, (int, float))} for row in subset])
        summary["aggregate"][variant] = {
            key: {"mean": float(np.mean([row[key] for row in subset])), "std": float(np.std([row[key] for row in subset], ddof=1)) if len(subset) > 1 else 0.0}
            for key in sorted(numeric_keys) if key not in {"seed"}
        }
    atomic_json(args.output, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
