# Qwen3-8B × UltraFeedback baseline verification

## Verdict

`qwen3_8b_ultrafeedback_margin_consumers_v1` is
`verified-for-controlled-UF` at the 1024-pair smoke gate. The next authorized
stage is validation-only hyperparameter selection; the 12 full seed runs must
not start before `selection.json` is frozen.

## Frozen inputs

- Model: Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`.
- UltraFeedback raw revision: `40b436560ca83a8dba36114c22ab3c66e43f6d5e`.
- H4 binarized revision: `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`.
- Final rows: train 36,258; validation 2,000; test 1,228.
- Train-only `q95=3.5`; train-only `mu_train=0.40212997328668354`.
- Non-thinking Qwen3 tokens: prompt 1024, completion 1024, total 2048,
  exactly one completion EOS.
- Durable reference cache: 39,486 unique finite pair-keyed rows; strict
  cached-vs-online maximum absolute error 0.0 at tolerance 1e-4.

## Formula and smoke gates

- Nine source/formula/data/collator tests and two launcher-resume tests passed.
- DPO, MMPO, ODPO log-gap, and Scaled-DPO normalized-gap transfer all finished
  16 optimizer steps with finite loss and gradients.
- All four used the same pair order, initial model, tokenized inputs, reference
  cache and effective batch size 64.
- DPO checkpoint 8 resumed to checkpoint 16 with model, optimizer, scheduler,
  RNG and Trainer state restored.
- Training had `ref_model is None`; no reference cache miss occurred.
- Each final model was merged, cast to BF16, checksummed, and only then had its
  large FSDP recovery shards removed.
- A separate non-result sanity run trained for two steps and evaluated 16
  validation pairs. It emitted the exact validation NLL, accuracy, reward,
  policy-reference log-ratio, and response-length keys consumed by pilot
  selection. Its temporary 92GB checkpoint was removed after verification;
  config, manifest, and metrics remain for audit.

## Resource decision

- Safe path: one 4-GPU FSDP1 FULL_SHARD task at a time, per-device batch 2 and
  gradient accumulation 8.
- The 2-GPU probe used per-device batch 1 and accumulation 32. At optimizer step
  2, `nvidia-smi` showed a maximum of approximately 81.059GB on one GPU, above
  the pre-registered 65GB safety limit. The process was terminated cleanly.
- Consequently `--parallel auto` is frozen to the four-GPU sequential fallback.
  This does not alter effective batch size, seeds, optimization, or data order.

## Remaining baseline work

1. Run the eight 128-step validation pilots (2 LR + 3 gamma + 3 alpha) and
   freeze LR/gamma/alpha.
2. Run four baselines at seeds 13, 42 and 100 for 566 optimizer steps.
3. Aggregate local validation/test diagnostics and freeze response artifacts.
4. Only after baseline results are complete, start `s_UF` calibration.
