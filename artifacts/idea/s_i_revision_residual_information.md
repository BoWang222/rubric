# 不枚举遗漏维度的 Rubric 充分性权重：Residual Preference Information

> **状态：历史调研材料（非执行规范）。** 本文件保留当时的候选方案与术语，可能与最终设计不同。正式实验只以根目录 [`PLAN.md`](../../PLAN.md)、[`CHECKLIST.md`](../../CHECKLIST.md) 和 [`final_experiment_matrix_audit.md`](../scout/final_experiment_matrix_audit.md) 为准；尤其不得从本文沿用旧称 “Scaled DPO”、旧 scorer/gate 或旧数据映射。

> **状态：已由 [s_i_rubric_function_final.md](./s_i_rubric_function_final.md) 取代。** 本文的 RPI 是依赖具体回答对和偏好标签的 pair-level 外部验证量，不再作为主 rubric-level \(s_i\)。

## 0. 修正后的结论

这项工作的核心不应写成“提出一种新的 robust DPO”。更准确的研究问题是：

> **现有 rubric 是对潜在真实偏好的一次有损测量。能否在不恢复、命名或补全任何遗漏评价维度的前提下，量化这次测量还遗漏了多少 preference-relevant information？**

robust learning 只是这个量的一个下游消费者：估计出的逐样本 rubric deficiency 可以进入 KL/Wasserstein ambiguity set，也可以用于过滤、主动复标、judge abstention 或 rubric auditing。这样，主要贡献落在 **rubric measurement insufficiency**，而不是已经较为拥挤的 robust preference optimization。

本轮检索后的主建议是：

1. 将 rubric 看成完整回答证据的一个**统计量/信息瓶颈**；
2. 用 **residual preference information（RPI）** 定义“给定 rubric 后，完整回答里还剩多少可用于预测理想偏好的信息”；
3. 数据集级用 conditional V-information 估计和检验，逐样本用两个校准后 preference posterior 的 KL/JS divergence 构造 \(s_i\)；
4. 不生成新的维度，也不声称找到了未测量因素 \(U\)；
5. 若没有独立于当前 rubric 的偏好标签或 full-context 判断通道，则充分性在统计上不可识别。Bayesian ensemble 可以提供一个有假设的替代，但不能凭空消除这个限制。

我建议把方法称为 **Residual Preference Information（RPI）**，把归一化后的充分性称为 **Rubric Information Retention（RIR）**。

---

## 1. 先区分三个容易混淆的问题

### 1.1 Informativeness：rubric 带来了多少有用信息？

它比较“有 rubric”与“没有 rubric”：

$
I_{\mathcal V}(R\to A)
=H_{\mathcal V}(A)-H_{\mathcal V}(A\mid R).
$

这个量大，表示 rubric channel 比 label prior 更能预测 preference。但“提供很多信息”不推出“提供了全部所需信息”。

### 1.2 Sufficiency：rubric 之外是否还残留有用信息？

它比较“只看 rubric measurement”与“在此基础上还能看完整回答证据”：

$
I_{\mathcal V}(H\to A\mid R)
=H_{\mathcal V}(A\mid R)-H_{\mathcal V}(A\mid R,H).
$

这个量才直接回答：**在当前 rubric 已经给定后，原始回答是否仍包含 rubric 没有保留的、可用于判断 preference 的信息？**

### 1.3 Robustness：当 nominal preference 可能错时，如何学习？

KL/Wasserstein minimax 是对不确定 preference distribution 的下游处理。它不定义 rubric 为什么不充分，也不识别遗漏信息。因此论文贡献链条应是：

$
\text{rubric as lossy measurement}
\rightarrow \text{RPI/RIR estimation}
\rightarrow \text{one downstream use: robust learning}.
$

近期 [PReMISE](https://arxiv.org/abs/2605.30803) 也把 preference fit、reliability 和 adversarial robustness 分成不同审计轴；[C2](https://arxiv.org/abs/2604.13618) 用 rubric 使 gold-label confidence 上升或下降来区分 helpful/misleading rubric。这两类工作测的是 rubric 的**帮助或匹配程度**，并没有给出“rubric 之外还剩多少偏好信息”的逐样本充分性量。

---

## 2. 不需要显式构造遗漏维度的理论定义

### 2.1 Rubric 不是维度清单，而是一个测量算子

记完整的可观察证据为

$
H_i=(x_i,y_i^A,y_i^B),
$

理想偏好标签为 $A_i^*\in\{0,1\}$。当前 rubric 为 $\mathcal C_i$。rubric evaluator 对该回答对产生的实际测量为

$
R_i=T_{\mathcal C_i}(H_i).
$

$R_i$ 不应只是 rubric 文本。一个合理实现是保留逐 criterion 的两答分差、证据和权重：

$
R_i=
\left\{
\left(
c_{ik},a_{ik},w_{ik},
v_\psi(x_i,y_i^A,c_{ik},a_{ik})-
v_\psi(x_i,y_i^B,c_{ik},a_{ik})
\right)
\right\}_{k=1}^{K_i}.
$

因此 $R=T_{\mathcal C}(H)$ 是 $H$ 的一个压缩或 garbling。问题不是“缺了哪个新维度”，而是这次压缩是否保留了完成 preference decision 所需要的信息。

### 2.2 充分性的理想条件

对 log loss、Brier score 等要求正确概率预测的 proper loss，rubric 充分当且仅当

$
P(A^*\mid H)=P(A^*\mid R)\quad\text{a.s.}
$

等价地，

$
A^*\perp H\mid R,
\qquad
I(A^*;H\mid R)=0.
$

这里完全没有定义或估计遗漏维度 $U$。所有未显式测量、但会在完整回答中留下 preference-relevant signal 的因素，都被边缘化进 $P(A^*\mid H)$。

这种定义有三条现成理论脉络支撑：

- **统计充分性 / Blackwell experiment comparison**：如果 $R$ 是 $H$ 的有损统计量，充分性问的是使用 $R$ 是否会增加最优决策风险；
- **Le Cam deficiency**：衡量从一个 statistical experiment 换成另一个 experiment 后的最坏决策性能缺口。[van Rooyen & Williamson, 2014](https://arxiv.org/abs/1402.4884)
- **loss-dependent Bayes sufficiency**：充分性取决于决策损失；0–1 loss 只要求保留 Bayes class，而 log loss/proper scoring rule 要求保留整个 predictive distribution。[Bayes-Sufficient Representations, 2026](https://arxiv.org/abs/2606.04045)

你的 $P_i^0=\operatorname{Bernoulli}(\bar p_i)$ 和 DRO 都作用在 preference probability 上，所以应该采用后者，而不能只用两位 judge 是否给出相同 hard label。

### 2.3 一个直接给出逐样本量的恒等式

因为 $R=T_{\mathcal C}(H)$ 是 $H$ 的函数，

$
\begin{aligned}
I(A^*;H\mid R)
&=\mathbb E_H\left[
D_{\mathrm{KL}}\left(
P(A^*\mid H)\,\|\,P(A^*\mid R)
\right)
\right].
\end{aligned}
$

因此自然的逐样本 rubric deficiency 是

$
d_i^{\mathrm{RPI}}
=D_{\mathrm{KL}}\left(
P(A^*\mid H_i)\,\|\,P(A^*\mid R_i)
\right).
$

它有三点正好对应你的需求：

1. 不需要列举隐藏维度；
2. 比较的是两个 preference distributions；
3. 平均后恰好是 residual conditional information，而不是一个任意 uncertainty heuristic。

这可以作为论文中的核心 proposition，证明只需要展开 conditional mutual information 的定义。

---

## 3. 对“当前 rubric 与 no-rubric 做 PVI 比较”的直接回答

答案是：**可以做，但必须明确比较方向和输入；普通的 rubric-vs-no-rubric PVI 只能测 helpfulness/informativeness，不能直接测 sufficiency。**

### 3.1 三个 prediction views

用完全相同的模型族、容量和训练协议，构造：

1. $f_0$：只学 label prior，不看样本；
2. $f_R$：只看 rubric-induced measurement $R_i$，不能看原始回答 $H_i$；
3. $f_H$：看完整 $H_i$，但不看 rubric；
4. 可选 $f_{RH}$：同时看 $R_i,H_i$，用于严格的 conditional V-information 实现。

“不能让 $f_R$ 看原始回答”非常重要。如果一个 rubric-conditioned LLM judge 同时看到回答全文，它可能暗中使用 rubric 没写出的标准。此时测到的是 judge 的 holistic 能力，不是 rubric measurement 的充分性。

### 3.2 普通 PVI 测到什么

标准 pointwise V-information 为

$
\operatorname{PVI}_R(i)
=\log \hat p_R(a_i^*\mid R_i)-\log \hat p_0(a_i^*).
$

它衡量 rubric channel 相对于空输入给这个 label 提供了多少 usable information。[Ethayarajh et al., ICML 2022](https://proceedings.mlr.press/v162/ethayarajh22a.html) 引入 PVI 来测单个样本相对给定模型族的难度；pointwise PVI 可以为负，因此不适合不经处理直接当 $[0,1]$ 权重。

如果所谓“有 rubric vs 无 rubric”是：两个 judge 都看 $H_i$，一个额外收到 $\mathcal C_i$，则

$
\log \hat p_{H,C}(a_i^*)-
\log \hat p_H(a_i^*)
$

衡量的是 rubric 的**增量帮助**。它不能区分：

- rubric 没用；
- no-rubric judge 已经内化了 rubric 中的原则；
- rubric 提供了一些新信息，但仍遗漏另一些决定性信息。

此外，仅用 rubric 文本 $\mathcal C_i$ 与空输入比较通常也不合理：一套完美 criteria 本身并不告诉模型这一对回答中谁更好，真正的测量对象应是 $R_i=T_{\mathcal C_i}(H_i)$。

### 3.3 正确的“反向 PVI”：测 rubric 之后还剩多少

对应充分性的 pointwise residual CVI 为

$
\delta_i^{\mathrm{CVI}}
=\log \hat p_{RH}^{(-fold)}(a_i^*\mid R_i,H_i)
-\log \hat p_R^{(-fold)}(a_i^*\mid R_i).
$

其跨样本均值估计

$
I_{\mathcal V}(H\to A^*\mid R).
$

这正是“rubric 给定之后，完整输入还能提供多少新 label-relevant information”。conditional V-information 已被用于 probing 和 free-text rationale evaluation；例如 [REV](https://aclanthology.org/2023.acl-long.112/) 测 rationale 在 input/label 之外增加了多少新信息。这里把方向翻转：测 raw evidence 在 rubric measurement 之外还残留多少 preference information。

数据集级的信息保留率可以写成

$
\operatorname{RIR}
=\frac{
H_{\mathcal V}(A^*)-H_{\mathcal V}(A^*\mid R)
}{
H_{\mathcal V}(A^*)-H_{\mathcal V}(A^*\mid H)
}.
$

也就是“rubric 捕获的 usable information / 完整回答可提供的 usable information”。rationale 文献中已有用空输入归一化 sufficiency 的思路，但也发现这种指标强烈依赖模型和 baseline，因此必须使用 matched models 与 held-out evaluation。[Carton et al., EMNLP 2020](https://arxiv.org/abs/2010.04736)

注意：RIR 比较适合作为数据集或 slice-level 指标。逐样本 PVI 的比值会因负值或很小分母而极不稳定，不建议直接当 $s_i$。

实际报告 RIR 时，应同时给出未归一化的 captured information、residual information、bootstrap confidence interval，并只在 full-context information 明显大于 0 时报告比值。有限样本和近似优化可能使经验 V-information 轻微违反理论上的 information ordering；不要用截断后的 $[0,1]$ 数值掩盖这种估计失败。

---

## 4. 推荐的逐样本 $s_i$

### 4.1 主版本：posterior discrepancy

用 cross-fitting 得到两个校准后的 preference distributions：

$
\hat P_i^H=\hat P(A^*\mid H_i),
\qquad
\hat P_i^R=\hat P(A^*\mid R_i).
$

为避免 KL 无界、方向不对称和概率接近 0 时爆炸，实际逐样本量建议用归一化 Jensen–Shannon divergence：

$
\hat d_i
=\frac{
D_{\mathrm{JS}}(\hat P_i^H,\hat P_i^R)
}{\log 2}
\in[0,1].
$

再定义

$
\boxed{
s_i=s_{\min}+(1-s_{\min})(1-\hat d_i)
}
$

解释：

- 两个 posterior 接近：rubric-induced measurement 保留了 holistic preference channel 的信息，$s_i$ 高；
- 两个 posterior 相差大：完整回答中存在当前 rubric 未保留的可用 preference information，$s_i$ 低；
- 不需要知道差异来自 factuality、安全、风格还是任何未命名因素。

如果希望更贴近理论 KL，可将

$
D_{\mathrm{KL}}(\hat P_i^H\|\hat P_i^R)
$

作为主分析量，而把 bounded JS 用于进入优化。二者分别承担“理论定义”和“稳定工程映射”。

### 4.2 为什么不把 confidence/entropy 乘进 \(s_i\)

假如两个 channel 都给出 \(P(A=1)=0.5\)，则 RPI deficiency 为 0：rubric 相对于完整信息没有额外损失，但这个样本本身不确定。这个“不确定”已经由 nominal preference \(\bar p_i\approx0.5\) 表示。

如果再将 margin、entropy 或 Fisher confidence 乘进 \(s_i\)，会重新混合两个量：

- \(\bar p_i\)：当前 pair 的 preference uncertainty；
- \(s_i\)：当前 rubric 相对于 full-context channel 的 information retention。

保持二者分离，更容易解释，也避免同一不确定性被重复计入 ambiguity set。

### 4.3 数据集级检验，而不只给一个分数

除了逐样本 \(s_i\)，论文还应给出总体充分性假设

\[
H_0:A^*\perp H\mid R.
\]

可以使用 predictive conditional-independence test，或 DIET 的 information residuals 做全局/分组检验。[Predictive CI testing](https://arxiv.org/abs/1908.00105)、[DIET, AISTATS 2023](https://proceedings.mlr.press/v206/sudarshan23a.html)

这会形成一个更完整的输出：

- RIR：总体保留多少信息；
- CI test：是否有证据拒绝 rubric sufficiency；
- \(s_i\)：哪些样本承担了最大的 residual information。

---

## 5. 最大的识别问题：偏好标签来自哪里？

这是本方法是否成立的分界线。

### 设置 A：有独立 human / holistic preference labels

这是最强设置。用不依赖当前 rubric 生成的 \(A_i^*\) 训练或评估 \(f_R,f_H\)，再进行 cross-fitting。此时 RPI 可以被解释为“相对于独立目标 preference 的 rubric information loss”。

### 设置 B：现有 label 正是由当前 rubric 生成

这会产生循环识别。若

\[
\tilde A_i=\mathbb 1\{w_i^\top R_i>0\},
\]

那么 \(R_i\) 当然可以很好预测自己的派生 label，即使它遗漏了真实偏好的关键因素。PVI、Fisher、Bayesian uncertainty 都无法修复这个逻辑问题。

因此，**不能用 rubric-generated labels 作为 rubric completeness 的唯一 gold target。** 它们可以训练 downstream DPO，但不能独立验证 \(s_i\)。

### 设置 C：没有 gold label，但可调用多个独立 holistic judges

可用一组冻结的、no-rubric、prompt-diverse/model-diverse holistic judges 产生 \(P_{H,1},\ldots,P_{H,J}\)，再用 Dawid–Skene / Bayesian classifier combination 类的小型概率模型推断 latent preference posterior。

这不必训练新的大网络：可以只估计每个 judge 的 confusion/reliability 和 latent label。无标签估计 classifier accuracy 与组合已有 Bayesian 方法。[Platanios et al., ICML 2016](https://proceedings.mlr.press/v48/platanios16.html)

但 LLM judges 往往共享训练数据、提示偏差和表面启发式，条件独立假设很可能不成立；已有工作表明无监督 ensemble 需要显式处理 classifier dependence。[Jaffe et al., AISTATS 2016](https://proceedings.mlr.press/v51/jaffe16.html)

因此这个版本必须命名为：

> **holistic-panel-relative rubric sufficiency**

而不能声称是真实 human-preference sufficiency。最好用一个小型独立人工 audit set 来校准并打破 latent-label permutation。

### 设置 D：既无独立 label，也无独立 full-context channel

此时 rubric completeness 不可从现有数据中识别。原因不是估计器不够强，而是两个世界在观测分布上可以完全一样：

- 世界 1：rubric 已充分；
- 世界 2：存在影响真实偏好的 \(U\)，但 \(U\) 没有进入任何观测 label 或独立判断通道。

在这种设置下，诚实做法是把 \(s_i\) 当 sensitivity parameter / partially identified interval，而不是声称对它进行 point estimation。这也正好说明 robust learning 的角色：**处理未识别残余的后果，而不是证明残余已经被测量。**

---

## 6. Fisher 与 Bayesian 到底能不能用

### 6.1 Fisher：不适合作为主定义

Fisher information 衡量的是在给定模型与观测变量已经指定后，数据对参数或 latent trait 的局部精度。IRT test information 同样回答某套题在某个能力区间测得多精确。它不能发现 measurement model 根本没有表示的 construct。

所以 Fisher 更接近：

- rubric evaluator 对已编码 criterion score 的精度；
- 当前 likelihood 下的 local curvature；
- parameter uncertainty。

而你的问题是 content/construct underrepresentation。高 Fisher 可以与严重遗漏同时出现。因此它只适合做“已测信号的 precision”消融，不适合命名为 rubric sufficiency。

### 6.2 Bayesian：可以不训练大网络，但不会自动解决识别

Bayesian 只是推断框架，不等同于“必须再训练一个 neural network”。可以采用：

- 对固定 judges 的 confusion matrices 放先验；
- Beta–Binomial/Dirichlet 后验；
- Bayesian classifier combination；
- 对 \(d_i\) 或 \(s_i\) 输出 credible interval。

它最有价值的作用是表达 finite-sample 和 judge-panel uncertainty：

\[
p(s_i\mid \text{panel predictions, audit labels}).
\]

但后验只在 prior、likelihood 与 judge-dependence 假设下成立。它不会从 rubric-generated labels 中创造独立的真实性信号。因此建议：**RPI 是 estimand；Bayesian ensemble 是无 gold 条件下的一种 estimator，而不是定义本身。**

---

## 7. 文献版图与真正的研究空位

| 文献线 | 已经解决什么 | 对本项目的借鉴 | 尚未解决什么 |
|---|---|---|---|
| [V-information / PVI](https://proceedings.mlr.press/v162/ethayarajh22a.html) | 给定模型族下，输入对 label 有多少 usable information；可到 pointwise | 构造 matched null/rubric/full predictors | 普通 PVI 是 informativeness，不是 sufficiency |
| [Conditional probing](https://aclanthology.org/2021.emnlp-main.122/) 与 [REV](https://aclanthology.org/2023.acl-long.112/) | 测一个变量在 baseline 之外新增多少 label information | 将方向翻转，测 \(H\) 在 \(R\) 之外的 residual information | 没有用于 rubric completeness 或 preference learning |
| [Rationale sufficiency](https://aclanthology.org/2020.acl-main.408/) 与 [normalized sufficiency](https://arxiv.org/abs/2010.04736) | 比较完整输入与 rationale-only 对预测的影响 | rubric measurement 可类比为选择性 rationale/压缩表示 | 指标模型依赖，且 pointwise calibration 不稳定 |
| [Blackwell/Le Cam deficiency](https://arxiv.org/abs/1402.4884) | 比较两个 statistical experiments 的决策信息损失 | 给“rubric 是有损测量”提供决策论定义 | 通常不是易直接估计的逐样本 weight |
| [Predictive CI](https://arxiv.org/abs/1908.00105) / [DIET](https://proceedings.mlr.press/v206/sudarshan23a.html) | 检验给定 \(R\) 后 \(H\) 是否仍预测 \(A\) | 正式检验 rubric sufficiency | 主要给 global test，不直接给稳定 \(s_i\) |
| [PReMISE](https://arxiv.org/abs/2605.30803) | rubric 的结构、可靠性、preference fit、adversarial robustness 审计 | 证明这些概念必须分开 | 未定义未测 preference information |
| [C2](https://arxiv.org/abs/2604.13618) | rubric 让 gold-label confidence 上升还是下降 | 强 helpfulness baseline | helpful rubric 仍可能不完整 |
| [Rubric decomposition/refinement](https://arxiv.org/abs/2602.05125) | 通过分解、过滤和加权改善 coverage | 说明 coverage 是现存 failure mode | 需要显式生成/补充维度，不满足本项目约束 |
| [Underspecification](https://www.jmlr.org/papers/v23/20-1335.html) | 多个同样拟合训练分布的模型在部署行为上可大幅不同 | Rashomon disagreement 可作辅助诊断 | 测的是模型 version-space，不等价于 rubric omission |
| [Bayesian unlabeled ensemble](https://proceedings.mlr.press/v48/platanios16.html) | 无 gold 时从多 classifier agreement 推断 accuracy/latent labels | no-gold holistic panel 估计 | 强依赖模型多样性和 dependence 假设 |
| [Sensitivity analysis](https://arxiv.org/abs/1507.03984) / [robustness value](https://academic.oup.com/jrsssb/article/82/1/39/7056023) | 不指定具体 \(U\) 的取值和维度，刻画多强的隐藏混杂才会改变结论 | 支撑“不识别 \(U\)，只量化结论对其敏感性”的哲学 | 给的是效应边界/全局敏感度，不是数据识别出的逐样本 rubric 信息量 |
| [Effect extrapolation](https://arxiv.org/abs/2102.01935) | 利用不同 measured-confounder 子集下的稳定性外推未测混杂影响 | 可类比为对已有 criteria 做随机 masking，只用作校验/辅助估计 | 依赖 criteria 记录/遗漏过程可外推的假设 |
| [Double negative controls](https://academic.oup.com/jrsssb/article/82/2/521/7056052) | 在 proxy、negative-control 和 completeness 假设下，不知道 \(U\) 的语义也可能检测/调整混杂 | 可启发 rubric-order、answer-swap、语义保持扰动等 falsification tests | 找到有效 negative-control exposure/outcome 本身需要结构知识，当前问题中通常不足以 point-identify \(s_i\) |

本轮大范围检索没有发现以下组合已经被直接解决：

> **将 instance-specific rubric 定义为 preference evidence 的统计压缩，在不枚举遗漏 criteria 的情况下，用 residual conditional usable information 得到逐样本 rubric sufficiency，并将其作为 preference-distribution trust parameter。**

因此创新点不必是凭空发明一个新 divergence；更可信的创新是把几条成熟理论正确拼接成一个此前缺失的 estimand、estimator 与验证协议。

因果敏感性文献也说明了一个重要边界：**不描述 \(U\) 的具体语义是可以的，但不对 \(U\) 或可观察 proxy 作任何假设则不可能。** Sensitivity analysis 可以完全不恢复 \(U\)，但它输出的是“多强的隐藏因素会推翻结论”；negative controls 可以不命名 \(U\)，但要有共享混杂机制的有效 proxy 与 completeness 条件。RPI 选择的是第三条路：不恢复 \(U\)，改为检测完整回答通道相对于 rubric bottleneck 的 residual predictive signal。它因此依赖“独立 preference target + full-context channel”，而不是依赖被枚举的新 criteria。

---

## 8. 建议的最终方法：RPI-Rubric

### 8.1 Estimand

\[
\mathcal I_{\mathrm{res}}
=I(A^*;H\mid R),
\qquad
d_i^{\mathrm{RPI}}
=D_{\mathrm{KL}}(P(A^*\mid H_i)\|P(A^*\mid R_i)).
\]

### 8.2 Practical estimator

1. 用 current rubric evaluator 得到 \(R_i=T_{\mathcal C_i}(H_i)\)；
2. 在独立 preference labels 上 cross-fit matched \(f_R,f_H\)；
3. 对 posterior 做 temperature/isotonic calibration；
4. 计算 \(\hat d_i=D_{\mathrm{JS}}(\hat P_i^H,\hat P_i^R)/\log2\)；
5. 映射成 \(s_i=s_{\min}+(1-s_{\min})(1-\hat d_i)\)；
6. 用 conditional V-information 和 CI test 验证 aggregate residual；
7. 将固定的 \(s_i\) 交给 KL/Wasserstein robust learner。

### 8.3 防止“两个模型差异”冒充“rubric 信息差异”

实验必须做到：

- 同一 backbone、参数规模、训练预算、label split；
- \(f_R\) 与 \(f_H\) 只改变信息通道；
- cross-fitting，不能在训练样本上计算 pointwise weight；
- 报告 Brier/ECE，posterior divergence 只有在概率校准后才有意义；
- answer A/B 交换、rubric paraphrase、judge family 替换；
- 多个 model seeds/families，报告 \(s_i\) 的方差；
- 用 \(f_{RH}\) 检查结论是否与严格 CVI 一致。

若 \(f_R\) 与 \(f_H\) 共享同一种 shortcut，它们可能错误地一致，使 \(s_i\) 虚高。因此 agreement 只能证明“相对于 full-context reference channel 的充分”，不能自动等于人类真值。独立 human labels/audit 仍是最强识别资源。

---

## 9. 实验设计：不在估计时找维度，但要能验证真的测到遗漏

### 9.1 主评估

在独立 human-preference test set 上检验：

- \(s_i\) 能否预测 rubric-only judge 的错误：AUROC/AUPRC；
- 按 \(s_i\) 从低到高拒答时的 accuracy–coverage / AURC；
- \(s_i\) 与 rubric-only error 的 calibration；
- 控制 nominal margin/entropy 后，RPI 是否仍有增量解释力；
- RIR、residual CVI 和 CI rejection 是否一致。

### 9.2 可控遗漏实验只用于验证，不用于估计

可以从具有较完整 annotation/rubric 的现有数据出发，随机或按重要性 mask 一部分 criterion，再检查 \(s_i\) 是否随 masking 增强而下降。

这里 estimator 从不被告知被 mask 的 criterion 是什么，也不生成补充维度；已知 mask 只作为实验者掌握的 ground truth。这样不会违背“不陷入寻找具体残余维度”的方法原则。

### 9.3 必须包含的 baselines

- rubric count/length/structural score；
- nominal margin、entropy；
- Fisher/Laplace predictive precision；
- rubric-vs-no-rubric confidence shift（C2-style helpfulness）；
- ordinary PVI \(f_0\to f_R\)；
- ensemble disagreement；
- RPI posterior KL / JS；
- global conditional V-information；
- uniform \(s_i\) 的 robust learner。

核心消融是：**ordinary PVI vs residual PVI/RPI**。它能直接证明“信息多”与“信息够”不是同一问题。

### 9.4 下游 robust learning 的正确位置

下游实验回答的是：一个更接近 rubric deficiency 的 \(s_i\)，是否比 uniform、margin、Fisher 等权重更能改善 corrupted/missing-rubric preference learning。

它是对 measurement 的效用验证，不是论文的唯一创新。论文标题、摘要与主方法部分都应先讲 rubric sufficiency；DRO 放在 downstream instantiation。

---

## 10. 推荐的论文 claim

较稳妥的英文表述：

> We formulate instance-level rubric adequacy as the loss-dependent sufficiency of a rubric-induced measurement of response pairs. Rather than recovering or enumerating latent criteria, we quantify unmeasured preference information by the residual predictive information in the full response evidence after conditioning on the rubric measurement. We estimate this deficiency through cross-fitted conditional V-information and calibrated posterior divergence, and demonstrate its use as a trust parameter in robust preference learning.

对应中文：

> 我们把逐样本 rubric 是否充分形式化为 rubric-induced measurement 对偏好决策的 loss-dependent statistical sufficiency。方法不恢复也不枚举潜在遗漏维度，而是测量在给定 rubric measurement 后，完整回答证据中仍残留多少可用于预测偏好的信息，并将其转化为逐样本 trust weight。

### 术语提醒

如果论文没有清楚的 treatment、outcome、backdoor path 和 causal estimand，直接称为 “unmeasured confounding” 可能被因果推断审稿人质疑。当前问题在统计上更精确的名字是：

- unmeasured preference-relevant information；
- omitted evaluative factors；
- rubric measurement insufficiency；
- latent preference confounding（若你给出明确结构模型）。

可以保留 unmeasured confounding 作为动机，但方法定义最好落在 measurement insufficiency / residual preference information 上。

---

## 11. 最终选择与决策边界

### 主方法

**选择 RPI：full-context vs rubric-bottleneck posterior discrepancy + aggregate conditional V-information。**

这是目前最符合四个约束的方案：

- 不列举未测维度；
- 明确比较两个 preference distributions；
- 有统计充分性和 conditional information 的理论基础；
- 能产生逐样本 \(s_i\) 并与当前 ambiguity set 对接。

### 次选方法

**Bayesian multi-judge latent preference posterior。** 当没有大量 gold label，但能够调用多种独立 holistic judge 时使用；它不是另训一个大网络，但必须承认模型相关性与相对真值问题。

### 不选作主方法

- 显式 rubric completion：违背 unmeasured premise；
- ordinary rubric-vs-null PVI：测 informativeness，不测 sufficiency；
- Fisher/IRT information：测已指定模型中的 precision，不测 construct undercoverage；
- raw entropy/margin：测 pair ambiguity，与 \(\bar p_i\) 重复；
- rubric 结构特征：测形式质量，不保证保留 preference information。

最重要的一句话是：

> **不要问“遗漏的维度是什么”，而要问“如果只允许使用当前 rubric 所诱导的测量，相比完整回答证据会损失多少对偏好决策有用的信息”。**
