# Experiment Logic Audit: Rubric Sufficiency and Robust DPO

> **状态：历史调研材料（非执行规范）。** 本文件保留当时的候选方案与术语，可能与最终设计不同。正式实验只以根目录 [`PLAN.md`](../../PLAN.md)、[`CHECKLIST.md`](../../CHECKLIST.md) 和 [`final_experiment_matrix_audit.md`](../scout/final_experiment_matrix_audit.md) 为准；尤其不得从本文沿用旧称 “Scaled DPO”、旧 scorer/gate 或旧数据映射。

## 1. The action point of `s(C)`

For pair `i`, the existing rubric system `C_g` measures a score gap

\[
m_{gi}=m(C_g;x_{gi},y^w_{gi},y^l_{gi}),
\]

which induces a nominal preference distribution, for example

\[
P^0_{gi}=\operatorname{Bernoulli}(p^0_{gi}),\qquad
p^0_{gi}=\sigma(\gamma m_{gi}).
\]

Because omitted preference factors are unmeasured, the method does not claim to know how to correct `m_i` directly. Instead, it permits the latent preference distribution `Q_i` to deviate from `P_i^0`:

\[
\mathcal U(s,\rho)=
\left\{Q:\sum_g\pi_g s(C_g)\frac1{n_g}\sum_i
D_{\mathrm{KL}}(Q_{gi}\|P^0_{gi})\leq\rho\right\}.
\]

`rho` remains one fixed constant. Therefore a larger `s(C_g)` implies a smaller allowed deviation, while a smaller `s(C_g)` implies a larger allowed deviation. `s(C_g)` changes the final minimax loss through the ambiguity set; it does not rescale, subtract from, or otherwise rewrite the observed gap `m_i`.

## 2. Required comparison rows

For every main margin consumer `B`, the clean comparison is:

1. vanilla binary DPO;
2. `B + rubric-derived m`;
3. `B + rubric-derived m + uniform robust (s=1)`;
4. `B + rubric-derived m + ours (s=s(C))`.

Row 3 is necessary to separate the value of the rubric-conditioned sufficiency statistic from the value of generic minimax robustness.

Main margin consumers: MMPO, ODPO, and Scaled DPO. ML-RDPO is an appendix extension.

## 3. Experiment tracks

### Track 0: identify `s(C)` without artificial rubric deletion

Do not construct 1D/2D/3D or nested rubrics. Estimate one `s_g` for each complete natural rubric system and verify its bootstrap, split-half, probe-seed, and grader stability. Use existing human-preference evaluation sets only as an independent diagnostic; no new annotation is collected.

Because a single-system constraint `s_g D_KL <= rho` is equivalent to a uniform radius `rho/s_g`, the main identification experiment must jointly train on multiple natural complete systems created by Auto-Rubric, OpenRubrics, and Rubric-ARM on the same prompt/response bank. Compare uniform, correct `s_g`, shuffled system-level `s_g`, and inverse `s_g` under one global fixed `rho`.

### Track 1: native-rubric plug-in evaluation

Use UltraFeedback and WildChecklists. Construct `m_i` from their native aspect/checklist scores, then compare MMPO, ODPO, and Scaled DPO under nominal, uniform-robust, and rubric-conditioned-robust objectives. WildChecklists/RLCF is a rubric-based data-construction pipeline rather than a distinct margin loss; its checklist gap should be passed to the selected margin consumer.

### Track 2: offline automatic-rubric evaluation

On one common prompt pool without native rubrics, hold fixed the sampled responses and pointwise grader. Generate rubrics with Auto-Rubric, OpenRubrics, or Rubric-ARM, turn criterion scores into winner/loser and `m_i`, calculate one system-level `s` per producer pipeline on a disjoint calibration bank, and compare nominal, uniform-robust, and ours. Native pairwise results and common pointwise-adapter results must be reported separately.

### Track 3: online/evolving rubrics

Treat Online Rubrics plus IterDPO as an optional extension. At round `t`, generate `C_t`, recompute one `s_t=s(C_t)` on the same fixed calibration bank, and train one DPO round. `s_t` may change across rubric versions but remains constant across samples within a round. `rho` remains unchanged across rounds.

## 4. Claim boundary

The method addresses uncertainty caused by systematic rubric incompleteness. A rubric-level scalar cannot identify the direction of an individual pair's error and should not be claimed to recover the true margin. The defensible claim is that it calibrates how much the optimizer should distrust the nominal preference distribution induced by the measured margins.

## 5. Audit of the existing execution plan

The existing plan is mathematically consistent on the following points: no additional SFT in the controlled main table; one shared policy/reference checkpoint; `s` is rubric-level rather than sample-level; `rho` is selected once and then fixed; uniform robustness is included; automatic-rubric producers share prompts, responses, and a common grader; and calibration/training/evaluation prompts are disjoint.

The main adjustment is ordering and emphasis: use complete UltraFeedback and WildChecklists rubrics for native-rubric transfer; use multiple natural complete automatic-rubric systems on one common response bank for the identification experiment; then run online rubrics last. Add vanilla DPO explicitly to the automatic-rubric track as a lower reference.
