# Rubric 可靠性权重 \(s_i\)：文献调研、方法选择与实验方案

> **状态：历史调研材料（非执行规范）。** 本文件保留当时的候选方案与术语，可能与最终设计不同。正式实验只以根目录 [`PLAN.md`](../../PLAN.md)、[`CHECKLIST.md`](../../CHECKLIST.md) 和 [`final_experiment_matrix_audit.md`](../scout/final_experiment_matrix_audit.md) 为准；尤其不得从本文沿用旧称 “Scaled DPO”、旧 scorer/gate 或旧数据映射。

> **状态：已被最终方案取代。** 本稿的显式 rubric completion / RCS 主线不符合“unmeasured factors 不应被恢复或枚举”的问题设定。请以 [Rubric-Blind Information Retention 最终方案](./s_i_rubric_function_final.md) 为准；本文件仅保留为第一轮调研记录。

## 0. 结论先行

这项工作的关键不应被表述为“根据置信度给 DPO 样本加权”，而应被表述为：

> **估计每个样本对“潜在遗漏 rubric 维度”的敏感性，并用该敏感性定义偏好标签分布的非均匀鲁棒集合。**

对 $s_i$ 的最佳主方案不是 raw Fisher information，也不是重新训练一个完整的 Bayesian reward model，而是：

1. 对当前 rubric 做**标签盲的反事实补全**，得到多个合理的候选遗漏维度；
2. 比较当前 rubric 与补全 rubric 下的偏好分布，测量**分布移动、决策翻转和评审稳定性**；
3. 用一个很小的独立人工审计集做 isotonic/logistic calibration；
4. 将校准后的“当前 rubric 仍可信”的概率映射为 $s_i$；
5. 在 KL/Wasserstein DRO 中让低 $s_i$ 样本更容易被对手移动。

我把这个估计器称为 **Rubric Completion Stability（RCS）**。Bayesian 最适合作为 RCS 的轻量不确定性层，例如对补全后的“有害翻转率”做 Beta–Binomial 后验，而不是再训练一个大网络。Fisher 可以作为辅助消融，但不宜作为主定义。

---

## 1. 先把 $s_i$ 的统计对象说清楚

记

$
H_i=(x_i,y_i^w,y_i^l),\qquad \mathcal C_i=\text{当前 rubric},
$

并令 $A_i^*$ 表示在充分评价维度下的潜在“正确”偏好。当前 rubric 下的 nominal preference distribution 为

$
P_i^0=\operatorname{Bernoulli}(\bar p_i).
$

这里有两个必须分开的量：

- $\bar p_i$：**给定当前 rubric 后**，当前评审器对偏好的信心；
- $s_i$：当前 rubric 本身是否涵盖了做出这个偏好所需的信息，即 nominal distribution 是否值得信任。

因此，不能只用当前 rubric margin、entropy 或 reward gap 来定义 $s_i$。这些量主要反映“这对回答难不难分”，并已进入 $\bar p_i$；再拿它们构造 $s_i$ 会把 pair difficulty、judge confidence 与 rubric completeness 混在一起，并形成重复计数。

更贴近目标的理想定义是：在可能存在的额外有效评价维度 $U_i$ 下，当前 rubric 是否仍保持偏好结论。若存在完整观测和真实标签，可以用残余条件信息表达：

$
I(A_i^*;U_i\mid H_i,\mathcal C_i)\approx 0
$

意味着额外评价维度不再改变决策，当前 rubric 相对充分。现实中 $U_i$ 未观测，因此 RCS 用“合理的 rubric 补全”作为对 $U_i$ 的可操作探针。

### 一个必须在论文中主动承认的识别限制

仅从 $(H_i,\mathcal C_i,\bar p_i)$ 无法无假设地识别 rubric 是否完整：未出现的评价维度既没有标签，也没有观测。至少需要以下一种外部信号：

- 独立的 full-context 人工审计；
- 多个具有差异性的评审器/角色；
- rubric 补全或删除形成的反事实扰动；
- 可控的合成 rubric missingness；
- 对潜在遗漏的结构性假设或敏感性分析。

这与 noisy covariates / unmeasured confounding 中通常只能做 partial identification 和 sensitivity analysis 的结论一致，而不是 Fisher 或 Bayesian 技巧能够自动消除的问题。[Guo et al., 2022](https://proceedings.mlr.press/v177/guo22a.html)、[Distributionally Robust Causal Inference](https://arxiv.org/abs/2210.08326)、[Sensitivity Analysis without Assumptions](https://arxiv.org/abs/1507.03984)

另一个直接推论是：若 $s_i$ 被限制为只依赖 $\mathcal C_i$，它至多是某类 rubric 的全局质量先验，不能判断该 rubric 对当前 prompt/answer pair 是否适用。真正的逐样本权重应至少允许

$
s_i=f(H_i,\mathcal C_i).
$

---

## 2. 你当前 DRO 中 $s_i$ 的位置是对的，但要澄清语义

你定义的加权距离

$
D_s(Q,P^0)=\frac1n\sum_{i=1}^n s_iD(Q_i,P_i^0)
$

会使高 $s_i$ 的样本更难被对手改变，低 $s_i$ 的样本更容易吸收 ambiguity budget。因此 $s_i$ 的准确语义是：

> **改变第 $i$ 个 nominal preference distribution 的局部代价/精度（local transport cost or precision）。**

它不是两个分布之间的距离本身。

### KL 情形

你的闭式解

$
q_i^*=\sigma\!\left(\operatorname{logit}(\bar p_i)+
\frac{\ell_i^+-\ell_i^-}{\lambda s_i}\right)
$

清楚显示：$s_i$ 越大，对手造成的 logit shift 越小。

### Wasserstein / Bernoulli 情形

因为

$
W_1(\operatorname{Bern}(q_i),\operatorname{Bern}(\bar p_i))
=|q_i-\bar p_i|,
$

内层问题等价于一个 fractional-knapsack 型分配：对手优先把预算给

$
\frac{|\ell_i^+-\ell_i^-|}{s_i}
$

大的样本。因此 $s_i$ 直接控制最坏分布先攻击哪些数据。

### 尺度不可辨识

若所有 $s_i$ 同乘常数 $c$，等价于改变 $\rho$ 的尺度。应固定一种可复现的标定：

- 保持 $s_i\in[s_{\min},1]$，固定 $s_{\min}$ 与概率到权重的映射，并在所有方法中共享；或
- 放弃上界 1，将权重归一化为 batch/dataset mean 1。

论文中不能同时任意调 $s_i$ 的整体尺度和 $\rho$，否则两者作用无法区分。

---

## 3. 候选方案比较

| 候选 | 实际测到什么 | 优点 | 关键问题 | 建议角色 |
|---|---|---|---|---|
| 当前 margin / entropy | 当前 rubric 下的 pair ambiguity | 免费、易实现 | 与 $\bar p_i$ 重复；高置信也可能遗漏关键维度 | 只作弱 baseline |
| Raw Fisher information | 给定似然模型下的局部参数曲率/可辨识性 | 有统计解释，可做局部近似 | 不测 omitted dimensions；语义可能反向 | 辅助消融，不作主方法 |
| 完整 Bayesian neural RM | 参数和预测后验不确定性 | 可区分部分 epistemic uncertainty | 成本高；仍受模型设定和共同偏差影响 | 非必要 |
| 轻量 Bayesian 校准 | 翻转率/失败率的后验和可信区间 | 无需新网络；可给保守下界 | 需要定义可观测的失败事件 | 推荐作为 RCS 的不确定性层 |
| PVI / conditional MI | rubric 提供的可用信息或遗漏后的残余信息 | 与“信息是否充分”最接近 | 通常需要 gold label 和两个预测视图/模型 | 有人工标签时的强 baseline |
| 结构审计 | atomicity、observability、scope、redundancy 等 | 易解释，适合找 rubric 缺陷 | 结构好不等于决策有效 | RCS 的辅助特征 |
| **反事实 rubric 补全稳定性（RCS）** | 加入合理遗漏维度后，偏好分布是否改变 | 直接针对 unmeasured rubric confounding；逐样本；可无训练实现 | 依赖补全器和 judge diversity，需要防止泄漏 | **主方案** |

---

## 4. 为什么 Fisher information 不是最佳主方案

假设 preference likelihood 是 Bernoulli logistic，logit 为 \(\eta\)，则对 \(\eta\) 的 Fisher information 为

\[
\mathcal I(\eta)=p(1-p).
\]

它在 \(p=0.5\) 时最大，而 \(p=0.5\) 恰好是偏好最不确定的位置。因此若直接用“大 Fisher = 高 rubric 可靠性”，方向甚至可能与直觉相反。对模型参数 \(\phi\) 而言，Fisher 还会乘上 \(\nabla_\phi\eta\nabla_\phi\eta^\top\)，主要反映当前模型在给定似然下对参数的局部敏感性。

Fisher 的适用前提是：观测模型和所含特征已经被正确指定。你的核心问题恰好是 rubric 中可能有未指定的评价维度，所以 Fisher 无法从原则上检测“模型里根本没有的变量”。它能告诉我们当前模型对已编码信号有多敏感，却不能证明现有 rubric 足够完整。

Fisher-based evidential learning 确实用 Fisher 几何重新加权不确定样本，但目标是给定分类模型下的 evidential uncertainty，不是验证评价标准的内容效度。[Deng et al., ICML 2023](https://proceedings.mlr.press/v202/deng23b.html)

可以保留两个 Fisher 消融：

1. \(s_i^{\text{Fisher}}\) 使用 inverse posterior variance / Laplace predictive precision；
2. 使用已有 reward model 的最后一层或 LoRA 子空间做低成本 Fisher/Laplace。

但它们应被明确称为“model uncertainty baseline”，而非 rubric completeness estimator。

---

## 5. Bayesian 是否本质上还是训练一个网络？

不是。Bayesian 的本质是对未知量放先验，并通过数据得到后验；未知量可以只是一个标量翻转率，而非神经网络权重。可选层级包括：

- Beta–Binomial：每个样本的补全后有害翻转率；
- Bayesian logistic / Bradley–Terry：偏好概率；
- hierarchical model：不同 rubric 类型、领域、judge 的共享先验；
- Gaussian process preference model；
- 在已有 RM 上做 post-hoc Laplace-LoRA，不重新训练主网络。

大模型语境中的 Bayesian RM 往往沿用神经 reward model，并通过 Laplace-LoRA 或 variational inference 近似后验；这是工程选择，不是 Bayesian 的逻辑要求。[Bayesian Reward Models for LLM Alignment](https://arxiv.org/abs/2402.13210)、[Bayesian Preference Learning for Test-Time Steerable Reward Models](https://arxiv.org/abs/2602.08819)、[Deep Bayesian Reward Learning from Preferences](https://arxiv.org/abs/1912.04472)

对你的问题，最划算的 Bayesian 用法是：**在 RCS 产生的多次反事实补全结果上，对不稳定率做后验收缩和保守下界**。

---

## 6. 推荐主方法：Rubric Completion Stability（RCS）

### 6.1 标签盲的反事实 rubric 补全

对每个 $(H_i,\mathcal C_i)$，使用冻结的 rubric proposer，从若干互补角色出发生成 $M$ 个候选补全：

$
\widetilde{\mathcal C}_{im}\sim
G(\cdot\mid H_i,\mathcal C_i),\qquad m=1,\ldots,M.
$

补全器只被要求提出：

- 当前 rubric 未覆盖、但对该任务可能改变判断的维度；
- 可从给定回答中观测和操作化的维度；
- 与已有 criterion 非同义重复的维度。

必须采用以下防泄漏措施：

- 不向 proposer 暴露 chosen/rejected 标签；
- 随机交换两个回答的展示顺序；
- 先提出遗漏 criterion，再独立评估偏好；
- proposer、judge 与被训练 policy 冻结或 stop-gradient；
- 尽量使用多角色或异构 judge，避免单模型自洽被误当成真实性。

### 6.2 获得基础和补全后的偏好分布

使用 $B$ 个 judge/rerun/paraphrase 形成集合分布：

$
P_i^0=\operatorname{Bern}(p_i^0),\qquad
P_{im}^+=\operatorname{Bern}(p_{im}^+).
$

这里 $P_i^0$ 是当前 rubric 下的 ensemble preference；$P_{im}^+$ 是加入第 $m$ 个候选遗漏维度后的 ensemble preference。

### 6.3 三个核心不确定性分量

**遗漏敏感性：**

$
u_i^{\mathrm{miss}}
=\frac1M\sum_{m=1}^M
\operatorname{JS}(P_i^0,P_{im}^+).
$

JS divergence 有界且对称，适合做估计量；也可用 Bernoulli Wasserstein
$|p_{im}^+-p_i^0|$ 作为消融。没有必要为了与内层 KL-DRO 一致而强行使用 KL；估计器与鲁棒集合可以采用不同散度。

**决策翻转风险：**

$
u_i^{\mathrm{flip}}
=\frac1M\sum_{m=1}^M
\mathbf 1\!\left[(p_i^0-1/2)(p_{im}^+-1/2)<0\right].
$

它专门捕捉“遗漏维度足以改变 preference”的事件。还可以加入尾部量

$
u_i^{\mathrm{tail}}=Q_{0.9,m}|p_{im}^+-p_i^0|
$

以免均值掩盖少数但致命的 criterion。

**测量不稳定性：**

$
u_i^{\mathrm{stab}}
=\frac1B\sum_{b=1}^B
\operatorname{JS}(P_{ib},\bar P_i),
$

由跨 judge、重复运行和 rubric paraphrase 的分歧构成。它测量“评价器是否可靠”，与 $u_i^{\mathrm{miss}}$ 的“rubric 是否有遗漏”保持区分。

### 6.4 从敏感性映射到 $s_i$

#### 最佳版本：小型人工审计集 + 低容量校准

抽取一小部分样本，由独立评审者在 full-context 下：

1. 补充关键遗漏 criterion；
2. 给出最终偏好；
3. 标注当前 rubric 是否遗漏了会实质改变决策的维度。

定义目标 \(R_i=1\) 表示当前 rubric-conditioned nominal preference 在完整审计下仍可信。用 cross-fitting 的 isotonic regression 或 logistic regression 校准：

\[
\widehat r_i=widehat{\Pr}\!\left(
R_i=1\mid
u_i^{\mathrm{miss}},u_i^{\mathrm{flip}},u_i^{\mathrm{tail}},
u_i^{\mathrm{stab}},u_i^{\mathrm{struct}}
\right).
\]

不需要神经网络；低容量校准更便于解释，也降低与主模型共同过拟合的风险。

若把二元偏好的随机水平 0.5 视为“无可靠信息”，推荐映射为

\[
s_i=s_{\min}+(1-s_{\min})[2\widehat r_i-1]_+.
\]

这样 \(\widehat r_i\le 0.5\) 时只保留最低运输代价，\(\widehat r_i=1\) 时取 1。若校准集出现系统性低于 0.5，应先纠正标签方向，而不是仅降低权重。

#### 无人工标签版本：Beta–Binomial 保守稳定度

定义一次“有害补全事件”

\[
F_{im}=\mathbf1\left[
\text{preference flip}
\ \lor\ |p_{im}^+-p_i^0|>\tau
\right].
\]

令该样本的有害补全率为 \(\eta_i\)，使用

\[
\eta_i\sim\operatorname{Beta}(a_0,b_0),\qquad
\eta_i\mid F_i\sim
\operatorname{Beta}\left(a_0+\sum_mF_{im},
b_0+M-\sum_mF_{im}\right).
\]

然后取稳定概率 \(1-\eta_i\) 的下侧可信界：

\[
s_i=s_{\min}+(1-s_{\min})
Q_{\alpha}\bigl(1-\eta_i\mid F_i\bigr).
\]

这个版本完全不需要训练新网络，但论文中应称其为 **rubric stability score**，而不是无条件声称它是真实可靠性概率。

### 6.5 结构审计只作为辅助

可为每个 rubric 加入以下可解释特征：

- atomicity：每条 criterion 是否只评一件事；
- operationalizability / observability：是否能从回答中验证；
- scope / applicability：是否与当前任务相关；
- consistency：criteria 是否冲突；
- effective dimensionality：是否真的提供多个独立判断轴；
- redundancy：criteria/score 的相关性和重复度。

这些特征可进入 \(u_i^{\mathrm{struct}}\)，但“结构合格”并不等于“能导向正确判断”，所以不能替代反事实补全和外部校准。

---

## 7. 信息论版本：最接近直觉，但更适合作强 baseline

V-usable information / pointwise V-information（PVI）衡量特定模型族能从输入中获得多少对标签有用的信息。标准形式为

$
\operatorname{PVI}(x,y)
=-\log g[\varnothing](y)+\log g'[x](y).
$

它可以发现难例和疑似错标样本，但“当前 rubric 相对 no-rubric 提供很多信息”并不代表 rubric 已经完整。与你的问题更匹配的是**残余 usable information**：

$
\operatorname{PVI}_{\mathrm{res},i}
=\log p(A_i^*\mid H_i,\mathcal C_i,\widetilde{\mathcal C}_i)
-\log p(A_i^*\mid H_i,\mathcal C_i).
$

若加入合理补全后仍能显著提升 gold-label likelihood，说明当前 rubric 信息不足。该方案需要 gold labels 或可信 full-context labels，通常还要两个预测视图，因此成本高于 RCS；in-context PVI 可以避免微调，但仍依赖上下文示例和标签。[Ethayarajh et al., ICML 2022](https://arxiv.org/abs/2110.08420)、[In-Context V-Usable Information](https://arxiv.org/abs/2310.12300)、[CMI/PID feature selection](https://www.jmlr.org/papers/v24/21-0482.html)

建议将 conditional PVI/CMI 放在“有人工审计标签”设置中作为理论感更强的对照，而不是无标签主方法。

---

## 8. 与最新工作的关系和真正可守住的新意

### 8.1 Rubric 质量文献

| 工作 | 与本项目的关系 | 需要吸收/区分的点 |
|---|---|---|
| [Rethinking Rubric Generation through Rubric Decomposition](https://arxiv.org/abs/2602.05125) | 指出 coverage、conflated dimensions、misalignment、redundancy，并做递归分解与相关性加权 | 其 signal-to-noise 分析支持“正向、非冗余维度越多越好”；你的增量应是逐样本遗漏敏感性 + DRO，而不是再做 rubric generation |
| [PReMISE](https://arxiv.org/abs/2605.30803) | 对 rubric-conditioned judge 审计结构充分性、可靠性、preference fit 和对抗稳健性 | 很接近“rubric 是否好”的测量框架；尤其要引用其观点：一致性/稳定性不等于有效性 |
| [C2](https://arxiv.org/abs/2604.13618) | 比较有/无 rubric 时 verifier 对 gold label 的 confidence shift | 可作为有 gold label 的直接 baseline，但主要测 rubric helpfulness，不直接测 completeness |
| [Multi-Role Rubric Generation](https://arxiv.org/abs/2607.01830) | 用多角色减少 dimensional blind spots | 与 RCS 的多角色补全高度相关；你的重点应是诊断和鲁棒几何，而非生成本身 |
| [OpenRubrics](https://arxiv.org/abs/2510.07743) | 对比式 rubric 生成，并通过 label consistency 筛选 | 可作为补全器/筛选器，但 label consistency 容易保留原标签偏见 |
| [Auto-Rubric](https://arxiv.org/abs/2510.17314) | propose–evaluate–revise，并用压缩原则形成 rubric | 可作为结构质量 baseline |
| [Rubrics as Rewards](https://arxiv.org/abs/2507.17746) | rubric-based reward 和 RL | 证明 rubric reward 的训练价值，但不解决逐样本 rubric omission |
| [JudgmentBench](https://arxiv.org/abs/2605.25240) | 在法律判断中，pairwise comparative judgment 显著优于 rubric 打分 | 是重要反例：rubric 更细并不自动意味着测量更有效 |

### 8.2 Robust / weighted DPO 文献

| 工作 | 与本项目的关系 | 新意边界 |
|---|---|---|
| [Distributionally Robust DPO](https://arxiv.org/abs/2502.01930) | 用 Wasserstein/KL ambiguity set 做 DPO | 不能声称“首次 robust DPO” |
| [DPO-PRO](https://arxiv.org/abs/2510.23590) | 逐 preference pair 的分布鲁棒优化，用软偏好分布处理噪声 | 与你的数学框架最接近；必须作为核心 baseline。区别必须落在 \(s_i\) 专门测 rubric omission，并改变 uncertainty-set geometry |
| [Aligner, Diagnose Thyself](https://openreview.net/forum?id=oIAUP1K5Dq) | 用 perplexity difference、DPO loss 和 token entropy meta-learn instance weights | 不能声称“首次逐样本可靠性加权 DPO” |
| [Uncertainty-Penalized DPO](https://openreview.net/forum?id=tYyeUbHiNe) | 用 reward model ensemble uncertainty 惩罚不确定 pair | RCS 必须证明超过 generic model uncertainty |
| [Robust Preference Optimization through Reward Model Distillation](https://openreview.net/forum?id=VajjTXRj6J) | 通过 RM soft signal 提升噪声稳健性 | 需区分 soft preference/noise 与 missing rubric dimensions |

### 8.3 推荐的论文 claim

不建议：

> We are the first to use reliability weights in robust DPO.

更可守住的是：

> We estimate instance-level **rubric omission sensitivity** through label-blind counterfactual rubric completion, and use it to induce a non-uniform ambiguity-set geometry for robust preference optimization.

也就是说，新意是“**遗漏维度的可操作估计器 + 该估计器在 DRO 几何中的作用**”这条组合链，而不是任何单独组件。

---

## 9. 实验设计：必须证明 \(s_i\) 测到的是遗漏，而不只是难度

### 9.1 可控 missingness benchmark（最重要）

从较完整的 rubric 开始，按以下方式删除 criteria：

- 随机删除；
- 删除高权重 criterion；
- 删除能改变 preference 的关键 criterion；
- 删除一个语义簇，避免只删同义项；
- 逐级增加 missing rate。

预期必须满足：

1. missing rate/importance 增大时 \(s_i\) 单调下降；
2. \(s_i\) 能预测补回 criterion 后的 label flip；
3. 控制原始 margin 后，\(s_i\) 仍有额外解释力；
4. RCS 显著优于 entropy、margin、Fisher 和 ensemble uncertainty。

### 9.2 小型真实人工审计

由专家补充遗漏维度并给 full-context preference。评估：

- 预测当前 rubric error / material omission：AUROC、AUPRC；
- 概率质量：Brier、NLL、ECE；
- selective prediction：risk–coverage curve、AURC；
- 与 expert omission severity 的 Spearman/Kendall correlation。

标准 conformal risk control 可以给总体选择风险的有限样本控制，但不能把 marginal guarantee 写成每个样本的 conditional guarantee。

### 9.3 下游 robust DPO

至少比较：

1. vanilla rubric-DPO；
2. 全局 KL/Wasserstein robust DPO；
3. DPO-PRO；
4. uniform \(s_i\) 的你的内层问题；
5. margin/entropy \(s_i\)；
6. Fisher/Laplace \(s_i\)；
7. ensemble/Bayesian uncertainty \(s_i\)；
8. PVI/CMI（有标签设置）；
9. RCS，无校准；
10. RCS + Beta–Binomial；
11. RCS + 人工 calibration（主方法）。

下游应同时报告 clean preference、随机 label noise、系统性 rubric omission 和 distribution shift。若只在随机翻转上有效，不能支持“解决 unmeasured rubric confounding”的主张。

### 9.4 关键消融

- 补全数量 \(M\)、judge 数量 \(B\)；
- 单角色 vs 多角色补全；
- 同模型 vs 异构模型 panel；
- 是否标签盲、是否交换回答顺序；
- JS vs Wasserstein vs flip-only；
- mean sensitivity vs tail sensitivity；
- calibration 方法与 audit-set size；
- \(s_{\min}\)、\(\rho\) 的二维敏感性曲面；
- online 更新 \(s_i\) vs 预计算/冻结 \(s_i\)。

### 9.5 训练泄漏与循环论证

- \(s_i\) 的 calibration split 与 policy training/evaluation 分开；
- 使用 cross-fitting 生成训练集权重；
- proposer 和 judge 不读取 chosen/rejected 标签；
- 不能只用同一个 rubric generator 自己补全、自己评估、再宣称 rubric 有效；
- 主结果至少包含一个人工或异构模型审计通道。

---

## 10. 最小可行实现（MVP）

第一版无需训练任何新的神经网络：

1. 使用现有 rubric evaluator 得到 \(p_i^0\)；
2. 用冻结 LLM、3 个互补角色为每个样本提出 \(M=6\) 个候选遗漏 criteria；
3. 过滤同义重复和不可观测 criteria；
4. 使用 2–3 个 judge，回答顺序各跑一次，得到 \(p_{im}^+\)；
5. 计算 \(u_i^{\mathrm{miss}},u_i^{\mathrm{flip}},u_i^{\mathrm{tail}},u_i^{\mathrm{stab}}\)；
6. 无人工标签时先用 Beta–Binomial lower credible bound 得到 \(s_i\)；
7. 固定 \(s_i\)，接入现有 KL/Wasserstein 内层优化；
8. 在小规模人工审计后，将 Bayesian heuristic 替换或融合为 cross-fitted calibration；
9. 首先完成可控 criterion deletion 实验，验证 \(s_i\) 确实响应 rubric missingness。

推荐主配置：

\[
u_i=\alpha u_i^{\mathrm{miss}}+
\gamma u_i^{\mathrm{flip}}+
\delta u_i^{\mathrm{tail}}+
\zeta u_i^{\mathrm{stab}},
\]

无标签初版可用

\[
s_i=s_{\min}+(1-s_{\min})\exp(-u_i),
\]

但正式主结果应优先使用人工校准后的 \(\widehat r_i\) 映射，因为手调 \(\alpha,\gamma,\delta,\zeta\) 的解释力和可复现性较弱。

---

## 11. 最终判断

- **Fisher：**不适合作为主 \(s_i\)。它测的是已指定模型内的局部信息/曲率，不是遗漏评价维度；保留为 uncertainty baseline。
- **Bayesian：**值得使用，但不需要再训练一个网络。最佳用法是给反事实补全的失败率做后验收缩和保守可信界。
- **信息论：**conditional PVI/CMI 在有可信 gold/full-context labels 时理论上很漂亮，适合作为强 baseline 或人工审计版扩展；无标签下不宜作为唯一方案。
- **最佳主方案：**标签盲、多角色的 rubric completion stability，加小型人工 calibration；它最直接对应“rubric 是否遗漏了会改变判断的维度”。
- **论文的新意：**不是 generic weighting 或 robust DPO，而是 **rubric omission sensitivity estimator → non-uniform ambiguity geometry → robust preference learning** 的完整链条。

这一定义也与当前算法直觉一致：rubric 越完整，补全引起的偏好分布距离越小，估计的可靠性越高，\(s_i\) 越大，对手越难移动该样本；rubric 越可能遗漏关键维度，\(s_i\) 越小，模型越需要对其保持鲁棒。
