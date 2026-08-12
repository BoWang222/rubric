# Current State Audit — 2026-08-12

## Intake Summary

- launch mode: continue existing experiment state
- user intent: reconcile the actual NUS-server state, update the checklist, and expose the next training route
- recommended next anchor: repair and verify the formal FSDP checkpoint path, then resume the seed-42 UltraFeedback baseline matrix

## Asset Matrix

| Area | Current asset | Trust level | Why | Missing proof | Recommended action |
| --- | --- | --- | --- | --- | --- |
| Environment | NUS `rubric`, 2×H100-80GB contract | trusted | smoke, pilots, save/resume diagnostic and 18 launcher/loss tests passed | formal checkpoint failed after a long run | keep the frozen training recipe; repair checkpoint infrastructure only |
| UltraFeedback data | v2 base/tokens/ref-logp cache | trusted | exact counts, q95, hashes, cached-vs-online and source-parity gates passed | none for baseline lane | reuse without regeneration |
| Baseline smoke | DPO/MMPO/ODPO/Scaled-DPO, 16 steps | trusted | shared pair order/cache and verification report | not a final result | reuse as execution gate |
| Development pilots | 5×64-step validation-only runs | trusted | all complete and metrics-only verification sealed | single pilot seed by design | freeze LR 1e-6, gamma 2.2, alpha 0.5 |
| Formal DPO seed 42 | step 0→283 partial run | usable with verification | finite loss/gradients and stable 13.6--14.3 s/step | checkpoint save failed; no test metrics or trusted checkpoint | classify as infrastructure failure and restart only after repair |
| Other UF baselines | MMPO/ODPO/Scaled-DPO seed 42 | missing context | no full-run directories or metrics | full training not started | run sequentially after DPO recovery |
| Other raw datasets | WildChecklists, HelpSteer2, Auto-Rubric, OpenRubric-v2, JudgmentBench, WildChat | usable with verification | revisions, core row counts and first-order schemas verified on NUS | production splits/tokens/ref caches mostly absent | prepare one dataset lane at a time after UF baseline/s_UF |
| Source repositories | nine pinned refs on NUS | reference only | exact local commits recorded | several model weights/prompts/provenance contracts still absent | do not install old baseline repos into `rubric`; use as read-only oracles |
| Robust methods | `robust.py` interface and PLAN contracts | reference only | direction is documented | KL/W1 solvers, rho contract, invariants and distributed tests incomplete | implement only after baseline results and `s_UF` |
| Git | `agent/enable-8gpu-training` | trusted | pushed commits and server cherry-picks are traceable | server hashes differ because of cherry-picks | continue syncing by source commit and file hashes |

## Reusable Assets

- Qwen3-8B model revision and non-thinking tokenizer contract.
- UltraFeedback processed data, tokenized parquet and durable reference-logp cache.
- Baseline source/formula parity tests and unified trainer.
- Frozen pilot selection: LR `1e-6`, MMPO gamma `2.2`, ODPO alpha `0.5`.
- Raw datasets already present on NUS; they must not be downloaded again unless a pinned revision changes.

## Conflicts / Unknowns

- `run_manifest.json` for the failed DPO full run still says `running`, although no process exists.
- `checkpoint-283` is an empty, unsealed partial directory and must never be selected for resume.
- A short save/resume diagnostic succeeded, while the long formal run failed at its first save. One instrumented retry is justified; repeated blind retries are not.
- The likely layer is checkpoint/runtime infrastructure, not baseline math, but the exact CUDA/NCCL cause is not yet proven.
- RubricARROW scorer weights, vLLM runtime, `s_UF` response bank and canonical rubric manifest are not yet ready on NUS.

Evidence paths on NUS:

- `/home/chunyuan/rubric/logs/full_seed42_2gpu.log`
- `/home/chunyuan/rubric/runs/baselines/qwen3_8b_ultrafeedback_margin_consumers_v1/full/dpo/seed_42/launcher.log`
- `/home/chunyuan/rubric/runs/baselines/qwen3_8b_ultrafeedback_margin_consumers_v1/selection.json`
- `/home/chunyuan/rubric/runs/baselines/qwen3_8b_ultrafeedback_margin_consumers_v1/smoke/smoke_verification.json`

## Route Recommendation

- next anchor: checkpoint-path repair under the experiment stage
- why now: all baseline/data gates before full training are trusted; the only immediate blocker is durable checkpointing
- what should not be repeated: UltraFeedback preparation, reference-logp caching, source-parity tests, smoke, or pilots
- what still needs verification: one instrumented DPO full retry, final checkpoint/test/merge, then the remaining three seed-42 baselines
