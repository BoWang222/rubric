# Qwen3-8B × UltraFeedback baseline implementation

This project implements one controlled comparison lane with four loss routers:

- `dpo`
- `mmpo`
- `odpo_loggap`
- `scaled_dpo_gap_transfer`

The legacy author repositories are read-only numerical oracles. They are not
installed into the `rubric` environment. Every run uses the same materialized
Qwen3 non-thinking tokens and the same pair-keyed reference-logp cache.

## Gates

1. `rubric-preflight` freezes the actual machine, package, source and model state.
2. `rubric-prepare-ultrafeedback` must reproduce all audited counts and hashes.
3. `pytest` must pass the FP64 source/formula parity fixtures.
4. `rubric-cache-reference` must produce complete, unique, finite token-sum logps.
5. `rubric-launch smoke` must finish all four 1024-pair runs; DPO must resume from step 8.
6. `rubric-launch pilots` freezes LR, gamma and alpha using validation only.
7. Only then may `rubric-launch full --parallel auto` expose official test metrics.

The current baseline stage deliberately does not implement KL-DRO or
Wasserstein-DRO. Only their shared protocol is reserved in `robust.py`.

## Server commands

Run these after `conda activate rubric` and
`cd /root/autodl-tmp/rubric`. Data preparation, reference caching, verification,
and smoke are idempotent after their complete manifests exist.

```bash
rubric-preflight --root /root/autodl-tmp/rubric

rubric-prepare-ultrafeedback \
  --raw-dir data/raw/ultrafeedback_raw \
  --h4-dir data/raw/ultrafeedback_binarized \
  --model models/qwen3-8b \
  --output data/processed/ultrafeedback/v2

rubric-verify-source-parity \
  --project-root /root/autodl-tmp/rubric \
  --output artifacts/baselines/qwen3_8b_ultrafeedback_margin_consumers_v1/source_parity.json

torchrun --nproc_per_node=4 -m rubric_dpo.cli.cache_reference \
  --dataset-dir data/processed/ultrafeedback/v2/tokens_qwen3_8b_non_thinking \
  --model models/qwen3-8b \
  --output data/cache/ultrafeedback_qwen3_8b_ref_v1

rubric-verify-reference-cache \
  --model models/qwen3-8b \
  --dataset-dir data/processed/ultrafeedback/v2/tokens_qwen3_8b_non_thinking \
  --reference-cache data/cache/ultrafeedback_qwen3_8b_ref_v1 \
  --output data/cache/ultrafeedback_qwen3_8b_ref_v1/verification.json

rubric-launch smoke --root /root/autodl-tmp/rubric
rubric-launch pilots --root /root/autodl-tmp/rubric
rubric-launch full --root /root/autodl-tmp/rubric --parallel auto
rubric-launch status --root /root/autodl-tmp/rubric

rubric-evaluate \
  --runs-root runs/baselines/qwen3_8b_ultrafeedback_margin_consumers_v1/full \
  --output runs/baselines/qwen3_8b_ultrafeedback_margin_consumers_v1/local_summary.json
```

`--parallel auto` reads the persisted probe verdict. On the current machine it
uses four-GPU sequential execution because the two-GPU probe failed the memory
safety gate. Re-running a phase skips completed runs and resumes incomplete
runs from their newest sealed checkpoint.

## Storage behavior

Full runs use checkpoints 283 and 566. A run is merged and recovery shards are
removed only after its manifest says `complete`, its metrics report step 566,
and the merged safetensors checksums have been written. `rubric-finalize-run`
can also be invoked manually for a single run.

Development pilots are metrics-only selection runs. After validation metrics,
config, manifest, and their hashes are verified, their large FSDP recovery
checkpoint is removed; pilot model weights are not merged or retained. Progress
is written to `pilot_progress.json`, final choices to `selection.json`, and a
background launch should write its outer log to `pilots_launcher.log`.
