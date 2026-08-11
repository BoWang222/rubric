# Preference-strength 与 rubric-DPO：baseline 和实验配置调研

> **状态：历史调研材料（非执行规范）。** 本文件保留当时的候选方案与术语，可能与最终设计不同。正式实验只以根目录 [`PLAN.md`](../../PLAN.md)、[`CHECKLIST.md`](../../CHECKLIST.md) 和 [`final_experiment_matrix_audit.md`](../scout/final_experiment_matrix_audit.md) 为准；尤其不得从本文沿用旧称 “Scaled DPO”、旧 scorer/gate 或旧数据映射。

## 结论

现有文献不能被简单地统称为“rubric-based DPO”。它们至少包含两个彼此独立的模块：

1. **strength producer**：用人工等级、LLM 评分、投票或 rubric/checklist 得到“赢了多少” \(m_i\)；
2. **strength consumer**：在 DPO 损失中把 \(m_i\) 用作 offset、soft target、sample weight 或 rating-matching target。

本文研究的 \(s_g=s(\mathcal C_g)\) 与 \(m_{gi}\) 不同：\(m_{gi}\) 是 rubric 对具体 pair 已经测到的胜负强度，\(s_g\) 是整套 rubric system 是否充分的 operational proxy。同一 dataset/rubric condition 中所有样本共享一个 \(s_g\)。最干净的论文定位因此是：**在已有 strength-aware/rubric-based preference optimization 上，用 rubric-level sufficiency 直接加权分布距离，同时保持 ambiguity budget \(\rho\) 为常数**。

不存在一个公开数据集，让 ODPO、MMPO、Rating-DPO、2D-DPO 和 RLCF 都已经按同一模型、同一训练数据报告了可直接横向比较的数字。最成熟的共同交集是 **UltraFeedback**，HelpSteer2 最适合 rubric omission，RLCF/WildChecklists 最适合真实 pipeline transfer。因此建议采用“共同 strength benchmark + 同数据多-rubric 受控验证 + 真实 rubric-DPO transfer”的三层设计。

## 增量模块并不绑定 RLCF

RLCF 是 **rubric/score/pair producer**，不是本文唯一能修改的训练算法。更一般地，对任意 strength-aware baseline (B)，只要能够写出偏好为正或负时的两支损失

\[
\ell_{gi}^{B,+}(\theta),\qquad \ell_{gi}^{B,-}(\theta),
\]

就可保留该 baseline 对 pair-level rubric gap (m_{gi}) 的原始使用方式，再统一包上本文的 preference-distribution robust layer：

\[
\min_\theta\max_{\{q_{gi}\}}
\sum_g\frac{\pi_g}{n_g}\sum_i
\left[q_{gi}\ell_{gi}^{B,+}+(1-q_{gi})\ell_{gi}^{B,-}\right]
\]

subject to

\[
\sum_g\frac{\pi_gs(\mathcal C_g)}{n_g}\sum_i
D_{\rm KL}\!\left(\operatorname{Bern}(q_{gi})\|\operatorname{Bern}(\bar p_{gi})\right)
\le \rho.
\]

其中 (\rho) 对所有 baseline、数据集和 rubric condition 均保持同一个预设常数。RLCF、UltraFeedback 或 HelpSteer-2D 决定 (\mathcal C_g,m_{gi},\bar p_{gi},s_g)；DPO、MMPO、ODPO 等决定 (\ell^{B,\pm})。因此“数据/测量系统”和“strength-consuming loss”可以正交组合。

### 与各 baseline 的兼容性

| baseline | 保留的原机制 | 本文插入位置 | 兼容性判断 |
|---|---|---|---|
| DPO | binary DPO logit | 对正/反偏好两支 DPO loss 做 KL-minimax | 直接；但若 nominal (\bar p_i\) 来自 score gap，应标为 rubric-soft DPO，而非纯原始 DPO |
| MMPO | (\bar p_i=\sigma(\gamma m_i)) 的 soft target | 把 MMPO 的固定 Bernoulli target 扩成其周围的 ambiguity set | **最自然的理论主底座** |
| ODPO / Margin DPO | (m_i) 作为 logit offset/target margin | 保留 offset，并对 outcome-conditioned margin loss 做最坏期望 | 直接，但需要对反向标签采用对称 loss |
| Scaled DPO | (m_i) 作为 sample weight | (w_i) 保留在两支 loss 上，外层再 robustify label probability | 直接 |
| RDPO | rating gap 平移 DPO logit | 保留 rating shift，再 robustify ranking outcome | 直接 |
| ML-RDPO | DPO ranking term + rating-gap square loss | 只 robustify label-sensitive ranking term；rating regression term保持原样 | 可行且语义最清楚 |
| MaPPO | rating/reward estimate 作为 prior | 保留 prior，再对 pairwise likelihood 的 label distribution robustify | 可行，作为第二梯队 |
| 2D-DPO | segment × aspect 多维 loss | 先聚合成 pair outcome loss 后 robustify | 可行但侵入性较高，不宜首轮主表 |
| RLCF + DPO | checklist scores 用于 pair mining，训练仍是 DPO | 在其 DPO loss 外加入本文模块 | **最自然的真实 pipeline transfer** |

### 不应冒充 rubric-strength baseline 的方法

- SimPO 的 margin 是全局固定超参数，不来自 rubric score gap。
- \(\gamma\)-PO、MWPO 的强度主要来自当前模型的 implicit reward/confidence，不能证明 rubric 中“赢多少”的信息被使用。
- Dr. DPO、KLDPO、DPO-PRO 是 robust controls。尤其 DPO-PRO 已直接对 preference distribution 做 DRO，是本文必须比较的最近邻；本文相对它的增量是 nominal probability 来自 rubric gap，以及用 dataset/rubric-level (s(\mathcal C_g)) 改变统一固定预算下的 ambiguity geometry。

### 识别要求

单个数据集、单个 rubric 下，(s_g) 是常数，约束在数值上可写成 (D_{\rm KL}\le\rho/s_g)。因此若只做一个 WildChecklists 条件，无法单独识别“(s_g) 的排序是否正确”；它只能证明模块可插入。主验证必须包含至少两个 rubric systems，并始终固定同一个 \(\rho)。UltraFeedback 的 1D/2D/3D/4D nested rubrics 正适合做这一识别实验；WildChecklists 用来做 pipeline transfer。

## 方法谱系

| 方法 | 如何使用“赢了多少” | 是否真的依赖外部评分/rubric | 适合本文的角色 |
|---|---|---:|---|
| DPO | 只使用 \(y_w\succ y_l\) | 否 | 必须的 binary baseline |
| ODPO | 在 DPO logit 中减去 pair-specific offset \(\alpha f(m_i)\) | 是 | 用户所说“减去偏置项”的直接对应方法 |
| Margin DPO | 在 log-sigmoid 内减去人工标注的离散强度 \(m_i\in\{1,2,3\}\) | 是 | HelpSteer2-Preference 原生 baseline |
| Scaled DPO | 用 \(m_i\) 乘整个 DPO loss，相当于强偏好重复采样更多次 | 是 | sample-weighting 路线的代表 |
| MMPO | 把 \(m_i\) 变成 soft target \(p_i=\sigma(\gamma m_i)\)，再做双向交叉熵 | 是 | 最适合作为本文 nominal preference \(P_i^0\) 的 baseline |
| Rating-DPO / ML-RDPO | 分别用 rating gap 平移 DPO logit，或同时匹配 ranking 与 rating gap | 是 | 2026 年最强、最直接的 rating-gap 对照；注意 RDPO 在这里指 Rating-DPO，不是 robust-DPO |
| 2D-DPO | 以 segment × aspect 分数调节 token/segment 的有效 \(\beta\) | 是 | 固定 rubric 的细粒度对照和 omission stress test |
| RLCF | rubric 分数先用于选出 preference pairs，原论文之后仍训练标准 DPO | 是，且为 instruction-specific checklist | 本文最重要的自然主载体 |
| SimPO | 使用固定 target margin \(\gamma\) | 否 | 固定 margin 控制；不能声称“知道每一对赢多少” |
| AlphaDPO / dynamic-margin 方法 | 从当前策略或隐式 reward 动态产生 margin | 否 | 排除“仅仅需要自适应 margin”的控制组，不是 rubric baseline |

### 关键原始文献

- [ODPO: Direct Preference Optimization with an Offset](https://arxiv.org/html/2402.10571)：明确指出 Bradley–Terry/DPO 只给偏好概率而不表达偏好成立的程度；定义 \(\Delta_r=\alpha f(\mathrm{score}_w-\mathrm{score}_l)\)。论文在 TL;DR 上使用人类 7 点 Likert 评分，并将评分差的函数作为 offset。
- [MMPO: Margin Matching Preference Optimization](https://aclanthology.org/2024.findings-emnlp.792/)：令 \(p_i=\sigma(\gamma m_i)\)，弱差异对应接近 0.5 的 target，强差异对应接近 1 的 target；在 UltraFeedback 与 SHP 上比较 DPO。
- [HelpSteer2-Preference](https://arxiv.org/html/2410.01257)：同一数据同时包含五维人类评分、偏好方向和 1/2/3 级偏好强度；论文给出 Regular、Margin、Scaled DPO。共有 7,118 个 pair，其中 train 6,766、validation 352。
- [Direct Preference Optimization with Rating Information](https://arxiv.org/html/2602.00603)：提出 Rating-DPO、Rating-IPO、ML-RDPO，并在 `ultrafeedback_binarized` 上统一比较 DPO、IPO、SimPO、DDPO、RPO、MAPPO。Rating-DPO 的损失是 \(-\log\sigma(\beta\Delta_\theta-\beta\Delta_{\hat r}/\beta_1)\)。
- [2D-DPO](https://aclanthology.org/2025.findings-naacl.455/)：使用 Helpfulness、Correctness、Safety、Completeness、Clarity 五方面的逐 segment 分数；Qwen2-7B-Instruct 和 Llama-3-8B-Instruct 上评测。
- [RLCF: Checklists Are Better Than Reward Models](https://arxiv.org/html/2507.18624)：公开 130k 条 instruction-specific WildChecklists、评分和训练代码；每个 criterion 由 Qwen2.5-72B-Instruct 评分，并保留差异最大的 40% response pairs 后训练 DPO。
- [SimPO](https://arxiv.org/html/2405.14734)：有固定 target margin，但不读取外部 \(m_i\)，因此只作为 fixed-margin control。

## 数据集选择

### A. 共同 strength benchmark：UltraFeedback

建议使用 [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) 的 `train_prefs`（约 61.1k rows，直接包含 `score_chosen` 和 `score_rejected`），同时保留原始 [UltraFeedback](https://github.com/OpenBMB/UltraFeedback) 的四方面评分：instruction-following、truthfulness、honesty、helpfulness。

它是最成熟的交集：MMPO、SimPO、Rating-DPO/ML-RDPO 等都在 UltraFeedback 家族上做过实验。主实验对所有方法使用完全相同的 pair，定义

\[
m_i=\frac{\mathrm{score}_{w,i}-\mathrm{score}_{l,i}}{9}\in[0,1].
\]

主表建议只用 \(m_i>0\) 的共同子集，避免把 score tie 强行解释为确定偏好；tie-inclusive soft-label 结果放附录。

四个方面对所有样本相同，所以完整 4D rubric 对整套数据只产生一个 \(s_{4D}\)，这符合本文定义。用原有方面构造 1D/2D/3D/4D rubric systems 后，每个 system 各有一个 \(s_g\)，可在相同 responses、相同固定 \(\rho\) 下验证 \(s_gD_{\rm KL}\) 的负相关约束。

### B. 人类 strength benchmark：HelpSteer2-Preference

[HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) 有 21,362 条 response 样本（约 10,681 prompts × 2 responses），五个 0--4 人类属性评分：helpfulness、correctness、coherence、complexity、verbosity。[HelpSteer2-Preference](https://arxiv.org/html/2410.01257) 进一步给同一 response pair 标注偏好方向、1/2/3 强度和文字理由。

这是检验 ODPO/MMPO/Scaled DPO/Rating-DPO 的最佳人类数据。固定五维 rubric 产生一个 dataset-level \(s_{5D}\)，不同 nested subsets 分别产生自己的 \(s_g\)。它最适合：

- 重现 Regular / Margin / Scaled DPO；
- 对比 rating gap 与直接人工 strength label；
- 进行 5D→4D→3D→2D→1D 的 controlled omission；
- 不需要新增任何人工标注。

### C. 真实 rubric-DPO transfer：RLCF / WildChecklists

若坚持“一个数据集一个 rubric”，则把 RLCF 的 checklist generator、criterion guidance、judge 与 aggregation rule 整体记为 \(\mathcal C_{\rm RLCF}\)，在 calibration prompts 上聚合得到一个 \(s_{\rm RLCF}\)。WildChecklists 中所有训练 pair 共享该值；逐 instruction checklist 只用于估计该系统的平均 operational information，不产生训练时的 sample-level \(s\)。

严格复用 RLCF：

- model/reference：Qwen2.5-7B-Instruct；
- response sampling：temperature 1.3，top-p 0.9；
- judge：Qwen2.5-72B-Instruct；
- 每个 criterion 原论文采样 25 次 0--100 judge score；
- score aggregation：importance-weighted checklist score；
- pair mining：保留至少一个 criterion 上差异最大的 40% pairs；
- training：2 epochs，global batch 1024，max length 2048，cosine LR 3e-6→2e-6；
- evaluation：IFEval、InFoBench、FollowBench、AlpacaEval 2、Arena-Hard。

在同一 pair 上统一定义 \(m_i=(S_{w,i}-S_{l,i})/100\)。不重新生成 rubric、不改变 chosen/rejected、不新增人工标注，只替换 loss 或添加 minimax sidecar。

### D. 受控遗漏：HelpSteer-2D

2D-DPO 的五方面逐 segment 分数可构造嵌套 rubric 子集。完整 5D 只在评估时作为 pseudo-oracle，不参与低维训练。原论文配置可作为复现锚点：Qwen2-7B-Instruct、Llama-3-8B-Instruct，\(\beta=0.2\)，0.1×SFT loss，700 steps，初始 LR \(10^{-7}\) cosine decay，8×A100-80GB；评测 Arena-Hard、AlpacaEval 2、MT-Bench。

## 推荐的最终实验矩阵

### Experiment 1：RLCF 上的真实 pipeline transfer

所有行使用相同 WildChecklists、相同 pair、相同模型和训练预算：

1. Qwen2.5-7B-Instruct（不训练）；
2. RLCF + DPO；
3. RLCF + ODPO；
4. RLCF + Scaled DPO；
5. RLCF + MMPO；
6. RLCF + Rating-DPO；
7. RLCF + MMPO + uniform-KL minimax；
8. RLCF + MMPO + ROIV-conditioned KL minimax（本文）。

这里第 3--6 行回答“收益是否只是因为利用了赢多少”；第 7 行回答“是否只是统一 robust learning”；第 8 行检验 rubric 函数 \(s(\mathcal C_{\rm RLCF})\) 对距离的加权是否有效。所有行使用同一个固定 \(\rho\)。RLCF 只有一个 \(s_{\rm RLCF}\)，不能在样本间 shuffle。

参数公平化：

- ODPO 用 \(\alpha m_i\)，在 validation 上搜索 \(\alpha\)；
- Scaled DPO 用 \(w_i=m_i/\mathbb E[m]\)，保证平均梯度尺度为 1；
- MMPO 用 \(p_i=\sigma(\gamma m_i)\)，在 validation 上校准 \(\gamma\)；
- Rating-DPO 统一搜索 rating trust \(\beta_1\)；
- uniform-KL 与 proposed 使用完全相同的常数 \(\rho\)；proposed 唯一增加的是距离前的 \(s(\mathcal C)\)。

### Experiment 2：UltraFeedback 上的共同算法 benchmark

建议使用 `HuggingFaceH4/mistral-7b-sft-beta` 作为 policy/reference；资源允许时增加 `allenai/Llama-3.1-Tulu-3-8B-SFT` 作为第二模型。比较：DPO、ODPO、MMPO、Scaled DPO、Rating-DPO、ML-RDPO、uniform-KL、本文方法。

完整四方面 rubric 产生一个 \(s_{4D}\)。进一步构造 1D/2D/3D/4D rubric systems，每个 system 对全部样本共享一个 \(s_g\)，在相同固定 \(\rho\) 下验证 \(s_gD_{\rm KL}\)。

评测：AlpacaEval 2 LC、Arena-Hard、MT-Bench；RewardBench/held-out preference accuracy 只作为 calibration 诊断，不替代 policy generation quality。

### Experiment 3：rubric omission stress test

在 HelpSteer-2D 或 HelpSteer2 上构造固定顺序与随机顺序两类 nested subsets：5D、4D、3D、2D、1D。对每个子集重新计算 \(m_i\)、nominal \(p_i\) 和 \(s(\mathcal C)\)，但完整 5D 分数只用于评估：

- \(s\) 与维度删减数量的 Spearman 相关；
- 子集偏好相对完整 5D 偏好的 flip rate；
- \(s_g\) 与各 rubric system 的 flip rate / score discrepancy 的 Spearman 相关；
- 不同遗漏程度下 DPO、MMPO、uniform-KL、本文方法的 policy 指标；
- 随机 label flip negative control，用来区分 rubric misspecification 与普通标签噪声。

## 训练与统计报告规范

- 首先单 seed 筛选全部方法；主表对 DPO、MMPO、uniform-KL、本文至少运行 3 seeds。
- 同一 experiment 中固定 model initialization、pair order、reference、max length、batch 和 optimizer；每个 baseline 获得相同 validation 调参预算。
- 主指标同时报告均值、标准差/置信区间与平均 response length，防止把 verbosity 当成质量提升。
- 不报告 rubric 内 \(m_i\) 与常数 \(s_g\) 的 pair-level correlation；跨 rubric systems 报告 \(s_g\) 与 preference flip、pseudo-oracle discrepancy 和 robust gain 的相关性。
- 不对 \(s_g\) 做数据集内 mean-one normalization，也不构造 rubric-specific budget。所有 rubric systems 使用同一个常数 \(\rho\)，差异只来自 \(s(\mathcal C_g)\) 与距离的乘积。

## 最小可行版本

如果计算预算有限，先只做：

1. HelpSteer-2D 的 5D/3D/1D rubric-level \(s_g\) 校准；
2. 对这三个 systems 比较 MMPO、uniform-KL 与 ROIV-KL；
3. RLCF + DPO；
4. RLCF + MMPO；
5. RLCF + MMPO + uniform-KL / ROIV-KL transfer。

ODPO、Scaled DPO、Rating-DPO 放入第二阶段。这五行已经能分别隔离 rubric feedback、measured strength、generic robustness 与 rubric-conditioned robustness。
