# Final experiment-matrix audit

Date: 2026-08-11

## Verdict

The method-centric matrix is the correct top-level organization, provided its columns are called **rubric systems/sources**, not datasets. The executable matrix has nine systems:

1. UltraFeedback native rubric;
2. WildChecklists/RLCF;
3. HelpSteer2 structured-feedback adaptation / human-magnitude anchor;
4. Auto-Rubric global Theme--Tips rubric;
5. OpenRubrics generator;
6. Rubric-ARM generator;
7. OnlineRubrics-local frozen-policy elicitation;
8. RRD-local (EvoLM authors' reimplementation);
9. frozen EvoLM official-checkpoint transfer.

Every `+ours` comparison requires a same-radius `uniform robust (s=1)` control. Cross-system absolute scores from different native datasets are not producer rankings; the defensible contrast is within-source `ours - uniform`. The matrix column is a source/producer, but the weight index is the exact criterion set. Canonical rubric JSON includes criterion wording and its bound guidance/anchors/direction/importance, and excludes prompt, response, producer, generator and grader. Identical hashes share one cached `s(C)`; different hashes are scored separately. Global rubrics use a common 500-prompt calibration population and yield one scalar, whereas prompt-specific rubrics use the frozen probe responses of their assignment prompt and yield rubric-level weights. The separate 500-prompt WildChecklists bank is now a cross-method distribution/parser/scorer diagnostic, not a device for collapsing every producer to one `s_g`.

The frozen main scorer remains reference-free [`OpenRubrics/RubricARROW-8B-Judge`](https://huggingface.co/OpenRubrics/RubricARROW-8B-Judge) with the [official probability-scoring extractor](https://github.com/Haoxiang03/RUBRIC-ARROW): freeze `top_logprobs=10`, read literal ` true`/` false` log-probs, and map `p_true-p_false` to `Z=clip((1+p_true-p_false)/2,0,1)`. A prompt-specific `s(C_i)` uses all frozen probe score vectors before chosen/rejected selection; chosen/rejected and margin are not inputs to `s`. Prometheus-7B-v2 remains only a fixed scorer sensitivity.

For every system, raw positive gaps are scaled only with that system's training split: `q95_g = numpy.quantile(raw_positive_gaps, .95, method="linear")` and `m=clip(m_raw/max(q95_g,1e-8),0,1)`. Raw ties are removed, normalized `m<0.05` is the preregistered near-tie filter, and validation/test reuse the train `q95_g`.

## Rubric granularity correction

| Source/producer | Rubric granularity | `s` contract |
|---|---|---|
| UltraFeedback | one fixed four-aspect rubric | one shared `s_UF` |
| WildChecklists | prompt-specific `requirements` | one `s(C_i)` per exact checklist; repeated checklists share cache |
| HelpSteer2 | one fixed five-attribute rubric | one shared `s_HS2` |
| Auto-Rubric main track | one Appendix-K global Theme--Tips rubric | one shared `s_AR`; query-specific release is a separate artifact diagnostic |
| OpenRubrics | prompt-specific generated rubric | one `s(C_i)` per generated criterion set |
| Rubric-ARM | prompt-specific generated rubric | one `s(C_i)` per generated criterion set |
| OnlineRubrics-local | prompt-specific `dedup(C0(x) union Ce(x))` | one `s_t(C_i)` per rubric per round |
| RRD-local | prompt-specific recursive rubric | one `s(C_i)` per recursive criterion set |
| frozen EvoLM | prompt-specific JSON rubric | one `s(C_i)` per generated criterion set |

## Verified method--data mapping

| System | Verified source/data | Reproduction status in this project | Native artifact contract | Controlled WildChat contract |
|---|---|---|---|---|
| UltraFeedback | [raw UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) 63,967 prompts + [`HuggingFaceH4/ultrafeedback_binarized`](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) `train_prefs` (~61.1k) | native released-data reconstruction | four nested aspect annotations; project recomputes margin and audits, rather than treats source `overall_score` as its margin | n/a |
| WildChecklists | [51,071-row official HF pair release](https://huggingface.co/datasets/viswavi/wildchecklists) + [RLCF code and separately linked intermediates](https://github.com/viswavi/RLCF) | native released-data reuse first; full scoring-pipeline reproduction separately gated | `requirements` plus aggregate gap `chosen_score-rejected_score`; criterion traces are not HF columns | n/a |
| HelpSteer2 | [21,362 rating rows](https://huggingface.co/datasets/nvidia/HelpSteer2) + [7,118-pair preference auxiliary release](https://arxiv.org/abs/2410.01257) | structured-feedback scalarization adaptation / human-magnitude anchor | project scalarization is adapted from NVIDIA RM model-card weights; human ±1/±2/±3 strength is validation only | n/a |
| Auto-Rubric | [paper](https://arxiv.org/abs/2510.17314), [38,459-row HelpSteer3-only HF release](https://huggingface.co/datasets/agentscope-ai/Auto-Rubric), [linked OpenJudge framework](https://github.com/agentscope-ai/OpenJudge) | official released-data check; Appendix-K HS3 global-rubric reuse + common-scorer adaptation on a new common-WildChat bank | released query-specific pairs provide direction but no project-ready continuous margin or verified generator checkpoint | published global rubric + common scorer supplies score/gap |
| OpenRubrics | [paper](https://arxiv.org/abs/2510.07743), [official code](https://github.com/wanghaoyu0408/OpenRubrics), [74,214-row OpenRubric-v2](https://huggingface.co/datasets/OpenRubrics/OpenRubric-v2), [RubricRM-v2 weights](https://huggingface.co/collections/OpenRubrics/rubricrm-v2) | official released-artifact reuse; common-WildChat adapter | released winner/judge trajectory is binary rather than a continuous margin | official generator + common scorer supplies score/gap |
| Rubric-ARM | [paper](https://arxiv.org/abs/2602.01511), [official code](https://github.com/wanghaoyu0408/OpenRubrics/tree/main/rubric-arm), [official checkpoints](https://huggingface.co/collections/OpenRubrics/rubricarm) | frozen official-generator transfer; no alternating-GRPO reproduction | no independent ARM dataset or final DPO-pair dump; both-order consistency can be reconstructed | frozen official generator + common scorer supplies score/gap |
| OnlineRubrics | [paper](https://arxiv.org/abs/2510.07284), [project page](https://labs.scale.com/papers/onlinerubrics) | local DPO adaptation only; original Generalist/Expert data, code and checkpoint unavailable | original current/control + API extractor/grader + GRPO cannot be faithfully reproduced from public artifacts | induction split freezes current/control policies; each common/calibration prompt uses `dedup(C0(x) union Ce(x))` from 8 paired auxiliary contrasts; the reference-free common scorer scores the candidate bank |
| RRD | [paper](https://arxiv.org/abs/2602.05125), [later EvoLM implementation](https://github.com/stellalisy/EvoLM/tree/main/scripts/configs/reward_mode) | WildChat-4K approximate local DPO adaptation; not original-author code | RRD-WU native binary satisfaction/aggregation yields its own pointwise score | frozen Qwen2.5-32B proposer, Qwen3-1.7B filter/item judge, and strong/weak references enable the recursive filter; RubricARROW `1[p_true>=p_false]` forms the binary matrix before WU is recomputed; label the row `RRD-local/common-binarized-WU adapter` |
| EvoLM | [paper](https://arxiv.org/abs/2605.03871), [official code](https://github.com/stellalisy/EvoLM), [official weight](https://huggingface.co/stellalisy/EvoLM-8B/tree/main) | frozen official-checkpoint transfer; not full co-evolution | official rubric structure + native frozen Qwen3-1.7B judge yields its own score | frozen EvoLM generates rubric text; the same common scorer used by the other five systems supplies pair/gap |

## Margin-consumer code status

| Consumer | Public implementation | Executable status in this project |
|---|---|---|
| DPO | [author repository](https://github.com/eric-mitchell/direct-preference-optimization) | use the unified current-TRL trainer; retain the author implementation as a numerical/reference check |
| MMPO | [author repository](https://github.com/kykim0/margin-matching-pref-opt) | public but tied to an older dependency stack; port the paper loss into the unified trainer and verify per-example numerical agreement |
| ODPO | [author repository](https://github.com/rycolab/odpo) | public, but its original entry/data path is not a generic chat-preference pipeline; add and explicitly label the common chat-data adapter |
| Scaled-BT-to-DPO adapter | no official Scaled-DPO implementation exists, because HelpSteer2-Preference publishes a Scaled Bradley--Terry reward-model objective | implement only the preregistered project adapter; never label it an official-code or paper reproduction |

## Loss compatibility correction

The current Bernoulli preference-label DRO has nominal distribution `Bernoulli(clip(sigmoid(gamma*m)))`; at zero radius it recovers the matching stabilized MMPO baseline, not original ODPO or the project's Scaled-BT-to-DPO adapter. If no target is clipped, the stabilized baseline equals original MMPO. Therefore:

- MMPO is the core preference-label DRO consumer.
- ODPO and the Scaled-BT-to-DPO adapter use a separately named outer loss-DRO wrapper centered on the empirical sample distribution, so zero radius recovers the corresponding frozen nominal loss.
- HelpSteer2-Preference publishes Scaled Bradley--Terry reward-model training, not Scaled DPO. The policy-loss analogue is an explicit project adaptation and must never be labeled a paper reproduction.
- Until the zero-radius tests pass, `ODPO+ours` and `Scaled-BT-to-DPO+ours` are not valid result labels.

MMPO weights each pair's Bernoulli KL contribution by its own `s(C_i)`. The ODPO/adapter interface uses a per-item weighted generalized KL with `phi(t)=t log t-t+1`; when all items share one rubric it reduces to scalar-`s` KL, while heterogeneous rubrics require the exact normalization-root dual. Both interfaces keep one global, dimensionless `rho`. The ODPO/adapter extension remains minibatch loss-DRO, not exact full-dataset DRO.

## Final reporting tiers

| Report | Systems | Rows | Seeds |
|---|---|---|---:|
| Core main table | all nine systems | producer-specific DPO; MMPO nominal; MMPO + uniform; MMPO + ours | 3 |
| Natural multi-system identification | six automatic producers on the common WildChat bank | joint MMPO nominal / uniform / correct-s / shuffled-s / rank-reversed-s | 3 |
| Global-rubric radius diagnostic | UltraFeedback, HelpSteer2, Auto-Rubric global | fixed-rho ours versus uniform robust with a validation-tuned scalar radius; oracle never updates global rho | 3 |
| Prompt-specific identification | WildChecklists and the five prompt-specific automatic producers | correct rubric-level `s` / uniform / within-producer shuffled / rank-reversed mapping | 3 |
| Consumer-transfer table | UltraFeedback, WildChecklists, Rubric-ARM, frozen EvoLM | ODPO and Scaled-BT-to-DPO adapter nominal / loss-DRO uniform / loss-DRO ours | 3 |
| Coverage appendix | remaining five systems | ODPO and Scaled-BT-to-DPO adapter nominal / loss-DRO uniform / loss-DRO ours | 1 |
| Method-native artifact checks | each method's released data/code setting | sanity and provenance only; no cross-method ranking | n/a |

Thus the user's wide screenshot remains the correct **run-coverage map**, after four corrections: title the columns `Rubric system / source`, add HelpSteer2 structured feedback, add a uniform-robust row, and rename `Scaled DPO` to the project's `Scaled-BT-to-DPO adapter`. It should not be printed as one homogeneous statistical table because some columns use native datasets while six automatic producers share a controlled WildChat bank, and the two loss-DRO consumers are secondary transfer interfaces.

## Remaining pre-run verification

This is a source-level verified plan, not a completed reproduction. Before full training, freeze exact repository commits, model/dataset revisions, split hashes, canonical rubric serialization/hash rules, probe-response banks, the RubricARROW contract, RRD local actors, Online induction bank, and the Auto-Rubric Appendix-K artifact. Run zero-radius, `s=1`, identical-rubric cache, and heterogeneous-`s` inner-solver tests; then freeze the one global `rho`. GPU model, memory and device count are intentionally not part of the research contract and must instead be recorded per run.
