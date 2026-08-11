# Auto-Rubric / OpenRubrics / Rubric-ARM implementation audit

> **状态：历史调研材料（非执行规范）。** 本文件保留当时的候选方案与术语，可能与最终设计不同。正式实验只以根目录 [`PLAN.md`](../../PLAN.md)、[`CHECKLIST.md`](../../CHECKLIST.md) 和 [`final_experiment_matrix_audit.md`](../scout/final_experiment_matrix_audit.md) 为准；尤其不得从本文沿用旧称 “Scaled DPO”、旧 scorer/gate 或旧数据映射。

Audit date: 2026-08-10. Only primary sources (official papers, official repositories, and official Hugging Face assets) are used below.

## Executive decision

All three methods are usable, but none natively releases a calibrated continuous winner–loser margin. Their native output is a rubric plus a pairwise preference verdict. Therefore:

- native binary-DPO reproduction: direct for all three (with different amounts of plumbing);
- MMPO / ODPO / Scaled-DPO reproduction requiring a continuous margin: use one shared pairwise-to-margin adapter;
- external commercial APIs: not required for the final comparison if released local checkpoints or open-weight backbones are used; they are required only to reproduce the original OpenRubrics curation pipeline exactly.

Recommended shared adapter: sample `M` responses per prompt, judge every unordered pair in both response orders, repeat `K` times if a soft confidence is desired, then fit prompt-local Bradley–Terry utilities. Select `argmax u` as chosen, `argmin u` as rejected, and set `m = u_chosen - u_rejected`. This preserves each method's native pairwise judge and avoids introducing a different pointwise scorer.

## 1. Auto-Rubric

### Identity

- Current paper title: **Auto-Rubric: Learning to Extract Generalizable Criteria for Reward Modeling**.
- arXiv: [2510.17314](https://arxiv.org/abs/2510.17314).
- The arXiv HTML for an earlier revision still displays **From Implicit Weights to Explicit Rubrics: A Training-Free Framework for Reward Modeling**. These are revisions of the same arXiv paper, not two baselines.

### Official assets and license

- Official dataset: [agentscope-ai/Auto-Rubric](https://huggingface.co/datasets/agentscope-ai/Auto-Rubric).
- No official GitHub implementation or official model checkpoint is linked from the paper or dataset card as of the audit date.
- Dataset license: Apache-2.0; the card notes that underlying source datasets retain their original terms.
- Released split: 38,459 train rows (about 290 MB). The current viewer exposes one source value, `helpsteer3_preference`.

### What is released

Each row contains conversation context, two candidate answers, the original chosen/rejected label, one or more query-specific rubric strings, rubric validity, and the refinement epoch. Thus the release already contains binary DPO pairs and rubric text.

The paper additionally learns a compact query-agnostic hierarchical Theme–Tips rubric set from HelpSteer3-Preference or UltraFeedback-Binarized. The released HF dataset is query-specific annotation data; it is not a packaged executable implementation of the local-induction/global-compression procedure.

### Native semantics

- Generates rubric text: yes, query-specific criteria followed by dataset-level Theme–Tips compression.
- Native judge: pairwise binary decision, explicitly formalized as `LLM_judge(y+ > y- | x, y+, y-, R)`.
- Pointwise candidate score: no.
- Released calibrated margin: no. Do not interpret source `preference_score` metadata as a rubric-derived winner–loser gap.
- Original rubric-induction sources: HelpSteer3-Preference and UltraFeedback-Binarized.
- Downstream policy experiment: Qwen2.5-7B-Instruct DPO on WildChat prompts labeled by a rubric-guided judge.

### Compute and API status

The paper uses Qwen3-32B as the default open-weight rubric-learning backbone and evaluates with Qwen3-8B/14B/32B/235B plus optional proprietary judges. Therefore an external API is optional, not intrinsic to the method. A 32B model in BF16 is tight on one H100 80 GB; 4-bit/AWQ or an 8B judge is the safe single-GPU implementation. The original DPO run used 16 H20 GPUs, so exact training throughput is not reproducible on one H100, but rubric induction from 70 pairs and local judging are feasible.

### Verdict

**Adapter needed.** Directly reuse the released rubric-annotated chosen/rejected pairs for the native binary-DPO baseline. For a continuous `m`, rerun rubric-guided pairwise judging with repeated, order-swapped decisions and convert to a Bradley–Terry margin. If new arbitrary prompts are required, implement the paper prompts and compression algorithm locally because official executable code is absent.

## 2. OpenRubrics / Rubric-RM

### Identity

- Paper: **OpenRubrics: Towards Scalable Synthetic Rubric Generation for Reward Modeling and LLM Alignment**.
- arXiv: [2510.07743](https://arxiv.org/abs/2510.07743).
- Official code: [wanghaoyu0408/OpenRubrics](https://github.com/wanghaoyu0408/OpenRubrics), especially [`openrubrics/`](https://github.com/wanghaoyu0408/OpenRubrics/tree/main/openrubrics) and [`rubric-rm/`](https://github.com/wanghaoyu0408/OpenRubrics/tree/main/rubric-rm).

### Official assets and license

- Code license: MIT.
- Dataset: [OpenRubrics/OpenRubric-v2](https://huggingface.co/datasets/OpenRubrics/OpenRubric-v2), 74.2k rows.
- Local rubric generators: [RubricRM-4B-Rubric-v2](https://huggingface.co/OpenRubrics/RubricRM-4B-Rubric-v2), [RubricRM-8B-Rubric-v2](https://huggingface.co/OpenRubrics/RubricRM-8B-Rubric-v2).
- Local judges: [RubricRM-4B-Judge-v2](https://huggingface.co/OpenRubrics/RubricRM-4B-Judge-v2), [RubricRM-8B-Judge-v2](https://huggingface.co/OpenRubrics/RubricRM-8B-Judge-v2).
- The 8B repositories contain full BF16 weights (about 16.4 GB each), not merely adapters.
- The HF dataset/model cards do not visibly declare an asset license. Do not silently infer it from the repository's MIT code license; confirm before redistribution. The code itself is clearly MIT.

### What is released

OpenRubric-v2 rows contain `instruction`, `response_a`, `response_b`, `winner`, `rubric`, a rubric-grounded judge trajectory, and `source`. These are immediately convertible to binary DPO (`chosen`, `rejected`). No new judging or API call is needed to train that native released-data baseline.

Source corpora include UltraFeedback, Magpie, Skywork-Preference (including HelpSteer2 and OffsetBias), Synthetic-IF, MegaScience, and Medical-o1. The release contains 18 source values.

### Native semantics

- Generates rubric text: yes; prompt-specific Hard Rules and Principles.
- Native judge: rubric-conditioned pairwise rationale plus final winner.
- Pointwise candidate score: no. The paper explicitly lists absolute scoring and multi-response ranking as future work.
- Released calibrated margin: no.
- Policy alignment: the paper already labels pairs with Rubric-RM and runs DPO on Qwen2.5-7B-Instruct.

### Compute and API status

There are two distinct routes:

1. **Exact original data curation:** official `openrubrics/` scripts call frontier services. The repository exposes Azure OpenAI (GPT/o4-mini), Google Gemini, and Amazon Bedrock DeepSeek backends; the paper reports GPT-4.1-Mini for rubric curation and Gemini-2.5-Flash-Lite for judging. Exact curation therefore requires API credentials and spend.
2. **Released local inference:** official `rubric-rm/` evaluation scripts serve the released Qwen3-4B/8B generators and judges through vLLM. This requires no commercial API and works on arbitrary prompts and response pairs.

Two 8B BF16 checkpoints total about 32.8 GB of weights and fit comfortably on one H100 80 GB, either simultaneously with controlled KV cache or sequentially. The 4B pair is the lower-risk throughput option.

### Verdict

**Direct reuse for rubric generation and binary pair labeling; adapter needed for `m`.** Prefer the released local RubricRM-v2 checkpoints, not the expensive original curation APIs. Use the common order-symmetrized pairwise-to-Bradley–Terry adapter to generate continuous margins for MMPO/ODPO/Scaled DPO.

## 3. Rubric-ARM

### Identity

- Paper: **Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training**.
- arXiv: [2602.01511](https://arxiv.org/abs/2602.01511).
- Official code: [`rubric-arm/` in wanghaoyu0408/OpenRubrics](https://github.com/wanghaoyu0408/OpenRubrics/tree/main/rubric-arm).

### Official assets and license

- Code license: MIT through the parent repository.
- Generator: [OpenRubrics/RubricARM-8B-Rubric](https://huggingface.co/OpenRubrics/RubricARM-8B-Rubric).
- Judge: [OpenRubrics/RubricARM-8B-Judge](https://huggingface.co/OpenRubrics/RubricARM-8B-Judge).
- Both are full Qwen3-8B BF16 checkpoints of about 16.4 GB each.
- HF model cards do not visibly declare a license; confirm before redistributing weights.
- There is no separate released policy-preference dataset advertised. The public code provides schema examples and expects user-supplied JSONL; the reward-model training source is the general-domain portion of OpenRubrics.

### Native semantics

- Generates rubric text: yes, prompt-specific Hard Rules and Principles.
- Native judge: pairwise binary decision with per-criterion explanations and a final `Winner:` line.
- Pointwise candidate score: no.
- Released calibrated margin: no.
- Training data: Stage-I warmup uses synthetic rubric/judge trajectories derived from UltraFeedback, Skywork, Magpie, and Synthetic Instruction Following; alternating RL uses non-overlapping portions of general-domain OpenRubrics.
- Native policy use: sample two responses, judge both response orders, retain order-consistent labels, and run standard DPO; iterative DPO repeats sampling, labeling, and training. Thus conversion to preference data is already part of the official method.

### Compute and API status

Using the released 8B generator and judge for inference and pair construction requires no external API and is straightforward on one H100 80 GB. Both models together use about 32.8 GB of BF16 weights.

Exact retraining of Rubric-ARM is not a one-H100 baseline. The official ms-swift recipe starts a reward server, rollout server, and distributed training job and its example allocation uses GPUs 0 through 7. Therefore use released checkpoints for the paper comparison; do not make full alternating-GRPO reproduction a prerequisite.

### Verdict

**Direct reuse for rubric generation and DPO pair construction; adapter needed for `m`.** This is the strongest implementation anchor because it already specifies dual-order filtering and downstream DPO/IterDPO. Extend its dual-order judge into repeated votes plus Bradley–Terry fitting to produce the margin required by the user's margin-aware DPO baselines.

## Common pairwise-to-margin adapter

For each prompt `x`:

1. Generate `M` candidate responses with the same frozen policy and decoding settings for all rubric producers.
2. Let producer `g` generate rubric `C_g(x)`.
3. For every pair `(a,b)`, query the native judge in both orders. Repeat each order `K` times if stochastic confidence is needed.
4. Drop or mark ties when the two orders are inconsistent. Estimate `n_ab` and `n_ba` from consistent votes.
5. Fit utilities with a regularized Bradley–Terry likelihood:

   `min_u - sum_{a<b} [n_ab log sigmoid(u_a-u_b) + n_ba log sigmoid(u_b-u_a)] + lambda ||u||_2^2`, with `sum_j u_j = 0`.

6. Define `winner = argmax_j u_j`, `loser = argmin_j u_j`, and `m = u_winner - u_loser`.
7. Cache rubric text, every raw judge trajectory, order, seed, vote count, fitted utilities, `m`, and the producer ID.

This adapter introduces no human labels, preserves the native pairwise semantics of all three systems, and provides the same numerical `m` contract to MMPO, ODPO, and Scaled DPO. A practical first pass is `M=4`, `K=2`; increase `K` only in robustness checks because judge calls grow as `2K * M(M-1)/2` per prompt.

## Recommended implementation order

1. Rubric-ARM released checkpoints: easiest end-to-end local pair-construction anchor.
2. OpenRubrics/RubricRM-v2 released checkpoints: equally local; useful SFT-only producer/judge contrast with ARM.
3. Auto-Rubric released HelpSteer3 pairs: native data baseline first; then implement paper prompts only if new-prompt generation is required.
