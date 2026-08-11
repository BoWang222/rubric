# Rubric 信息权重的旧版修订：rubric-level blind-spot audit（已被 text-only RSIV 方案替代）

> **状态：历史调研材料（非执行规范）。** 本文件保留当时的候选方案与术语，可能与最终设计不同。正式实验只以根目录 [`PLAN.md`](../../PLAN.md)、[`CHECKLIST.md`](../../CHECKLIST.md) 和 [`final_experiment_matrix_audit.md`](../scout/final_experiment_matrix_audit.md) 为准；尤其不得从本文沿用旧称 “Scaled DPO”、旧 scorer/gate 或旧数据映射。

> **状态：superseded。** 用户后续明确要求：$s_i=s(\mathcal C_i)$、不使用人工或 holistic preference labels，并优先采用简单统计指标。当前主版本见 `s_i_semantic_information_volume.md`。

## 0. 最终结论

这次修订做四个关键更正。

1. 旧稿中的 \(A\) 只是“回答 A 是否应获胜”的二元标签，\(H_{\mathcal V}\) 是某类预测器的最小 log-loss。那个 PVI/RPI 定义依赖回答对和偏好标签，因此不适合作为本文要求的 **rubric-level** \(s_i\)。它应从主方法撤下，只作为外部验证。
2. 必须分开：rubric 对回答的分数、两个回答的分差、rubric 本身的充分性。这三个量分别是

   $
   r_{\mathcal C_i}(x_i,y),\qquad
   m_i=r_{\mathcal C_i}(x_i,y_i^w)-r_{\mathcal C_i}(x_i,y_i^l),\qquad
   s_i=s_{\Pi}(\mathcal C_i;x_i).
   $
3. 标准 DPO 只接收 chosen/rejected；rubric 可以先产生连续分数，但主流 rubric-to-DPO pipeline 通常在训练前把它压成二元 pair。若要保留“赢多少”，应使用 MMPO 式 soft preference 或 ODPO/rating-DPO 式 target margin。
4. 任何固定 rubric 文本都不可能凭自身证明“我没有遗漏未知因素”。因此主指标不应声称是无条件的绝对 completeness，而应定义为：**在一套固定、rubric-blind 的审计分布下，当前 rubric 保留了多少整体偏好信号**。

推荐主方法称为 **Rubric-Blind Information Retention（RBIR）**。它不恢复、不命名、不补全遗漏维度；它只搜索和统计“整体偏好明显变化、而当前 rubric 没有感知到”的 blind spots。

---

## 1. 旧式中的 $A$ 和 $H_{\mathcal V}$ 到底是什么

旧稿写过

$
I_{\mathcal V}(R\to A)
=H_{\mathcal V}(A)-H_{\mathcal V}(A\mid R).
$

其中：

- $A\in\{0,1\}$ 是偏好标签；例如 $A=1$ 表示第一个回答应获胜，$A=0$ 表示第二个回答应获胜；
- $R$ 是 rubric 对某个回答对产生的测量结果；
- $H_{\mathcal V}(A)$ 是模型族 $\mathcal V$ 在不看 $R$ 时预测 $A$ 的最小平均 log-loss；
- $H_{\mathcal V}(A\mid R)$ 是允许模型看 $R$ 后的最小平均 log-loss。

两者之差就是：看到 rubric measurement 后，预测胜负标签的损失下降了多少。

这一定义本身没有错，但它回答的是“rubric measurement 对标签预测有多有用”，不是“某个 rubric 文本本身表达了多少信息”。它需要一批回答、标签和预测器，因此用户对旧方法的质疑成立。后文不再用 \(A,H,R\) 定义主 \(s_i\)。

---

## 2. 用户关于“必须是 rubric 的函数”的判断：哪里正确，哪里需要修正

### 2.1 正确的部分

如果 \(s_i\) 的语义是“第 \(i\) 个 rubric 的质量或信息充分性”，那么它不应该由当前训练 pair 的难度决定。也就是说，主方法不应写成

$
s_i=s(x_i,y_i^w,y_i^l,\mathcal C_i).
$

否则，同一个 rubric 面对一个容易 pair 得到高 \(s\)，面对一个困难 pair 得到低 \(s\)；这混合了 **rubric quality** 与 **pair difficulty**。

主定义应是

$
s_i=s_{\Pi}(\mathcal C_i;x_i),
$

其中 $\Pi$ 是预先固定的 audit protocol。若 instance-specific rubric 已经把 prompt、reference 和适用范围完整写入 $\mathcal C_i$，可简写成 $s_{\Pi}(\mathcal C_i)$。

若多条样本使用完全相同的 rubric，而且坚持严格的 $s(\mathcal C)$，这些样本必然得到相同权重；任何逐 pair 变化都只能来自 $x$、回答对或额外随机信息。本文之所以仍可得到 instance-level $s_i$，是因为 $\mathcal C_i$ 本身是 instance-specific，或者更严谨地使用 $s_{\Pi}(\mathcal C_i;x_i)$。

### 2.2 不能省略的部分

若 rubric 不包含任务 $x_i$，则仅看 rubric 文本无法判断它是否与任务相关。例如同一套“文风优美、长度适中”的 rubric 对写诗可能相关，对数学证明可能严重不充分。因而：

- 纯 $s(\mathcal C)$ 能测结构清晰度、非冗余度、潜在评分容量；
- 任务充分性至少是 $s(\mathcal C,x)$；
- 对潜在真实偏好的绝对充分性还需要一个外部目标或审计分布。

这不是估计器能力不足，而是可识别性限制。一个固定文本没有随机性，形式上的 $I(\mathcal C;A)=0$；若 $\mathcal C_i$ 随任务变化，$I(\mathcal C;A)$ 又可能主要学到任务身份，而不是 rubric 质量。

### 2.3 “函数”不等于“不需要校准数据”

测量学中，试卷可靠性是试卷的属性，但必须让一批受试者作答才能估计；相同地，$s_{\Pi}(\mathcal C)$ 是 rubric 在固定审计协议 $\Pi$ 下的 functional，但可通过独立校准样本估计。

关键不是完全禁止 $y$，而是：

- 不使用当前 DPO 训练 pair 来定义 $s_i$；
- audit responses 由固定、与 rubric 隔离的协议生成；
- 对同一 $\mathcal C_i$，audit 完成后得到一个固定 $s_i$，再用于所有相关训练 pairs。

### 2.4 一个最短的不可识别性证明

固定同一个 rubric $\mathcal C$，考虑两个潜在真实效用世界：

$
U_1(y)=r_{\mathcal C}(y),
\qquad
U_2(y)=r_{\mathcal C}(y)+\lambda U_{\rm miss}(y).
$

世界 1 中 rubric 完全充分；世界 2 中 $U_{\rm miss}$ 会改变偏好而 rubric 不充分。但任意纯文本函数 $f(\mathcal C)$ 在两个世界取值完全相同。因此没有外部任务/行为证据时，纯 $f(\mathcal C)$ 无法区分这两个世界。本文选择估计相对于固定 audit protocol $\Pi$ 的 $s_{\Pi}(\mathcal C;x)$，而不是伪装成无条件真值。

---

## 3. Rubric-DPO 实际如何构造偏好数据

用户的理解“rubric 是为了从谁赢扩展到赢多少”只对了一半。Rubric 的确能产生分项分数和总分差，但它在现有工作中还有 grounding judge、暴露评价依据、过滤噪声 pair、挑选大间隔 pair 等用途；而且大多数 pipeline 最终仍把连续分数压成 chosen/rejected。

| 工作 | Rubric / 标签来源 | 分数如何构造 pair | 训练 loss 是否保留“赢多少” |
|---|---|---|---|
| [标准 DPO](https://arxiv.org/abs/2305.18290) | 人或 AI 直接给 chosen/rejected | 输入就是 \((x,y^w,y^l)\) | 否；没有外生 pair strength |
| [UltraFeedback](https://arxiv.org/html/2310.01377) | GPT-4 从 instruction following、truthfulness、honesty、helpfulness 四方面给 1–5 分和 critique | 后续 Zephyr 等工作从评分/排序得到 pair | 在普通 DPO 中否 |
| [RLCF / Checklists](https://arxiv.org/html/2507.18624) | 自动 checklist；Qwen judge 与程序 verifier；每项 0–100，多次采样取均值 | importance-weighted score；只保留差异最大的 40% pairs | 否；最终两轮标准 DPO |
| [Visual rDPO](https://arxiv.org/html/2604.13029) | 每个 image-instruction 生成 essential/additional rubric；VLM judge | 每项 0/0.5/1，权重 1/2/3；设 margin threshold，并选最大 reward-margin pairs | 否；margin 用于挖 pair，训练仍是 DPO/MPO |
| [OpenRubrics](https://arxiv.org/html/2510.07743) | 混合 AI score、verifier 和已有人工偏好；生成 rubric 后做 label-consistency filtering | rubric-conditioned judge 复现 inherited preference | 主要仍是二元 verdict |
| [Configurable Preference Tuning](https://arxiv.org/html/2506.11702) | teacher 按 low/high target rubric level 合成回答 | 按 target level 构造方向相反的二元 pairs | 否；target level 进入生成，不进入标准 DPO loss |
| [2D-DPO](https://aclanthology.org/2025.findings-naacl.455/) | GPT-4 对 segment × aspect 打分，部分人工核验 | 保留多维、分段监督 | 是；分数直接进入优化，是少数例外 |
| [Rubrics as Rewards](https://arxiv.org/html/2507.17746) | 自动/人工 grounded rubric，LLM judge | 加权 criterion score 直接形成 scalar reward | 是，但使用 GRPO，不是 DPO |

所以 rubric-based preference data 的来源可以是：

1. 人工在 rubric 指导下直接选 A/B；
2. 人工逐维打分，再按总分排序；
3. LLM judge 逐维打分，再按总分选 highest/lowest 或设 margin threshold；
4. rubric-conditioned judge 直接输出 A/B；
5. teacher 按高/低 target level 合成天然的正负回答。

近年的 instance-specific rubric pipelines 主要依靠 AI judge 大规模打分，人工通常用于 rubric/reference grounding、验证子集或最终 benchmark，而不是逐对全量标注。

### 3.1 标准 DPO 的连续 margin 不是外生的“赢多少”

标准 DPO 内部有连续 log-ratio margin

$
z_i(\theta)=\beta\left[
\log\frac{\pi_\theta(y_i^w\mid x_i)}{\pi_{\rm ref}(y_i^w\mid x_i)}
-\log\frac{\pi_\theta(y_i^l\mid x_i)}{\pi_{\rm ref}(y_i^l\mid x_i)}
\right],
$

但这是 policy 当前学出来的 implicit margin，不是 annotator/rubric 告诉它的真实 quality gap。普通 loss

$
-\log\sigma(z_i)
$

只知道方向 $y_i^w\succ y_i^l$。

### 3.2 真正保留“赢多少”的近邻

- [ODPO](https://aclanthology.org/2024.findings-acl.592/) 把外部 score gap 作为 pair-specific target offset；
- [MMPO](https://aclanthology.org/2024.findings-emnlp.792/) 将质量差映射成 Bradley–Terry soft target；
- [HelpSteer2-Preference](https://arxiv.org/html/2410.01257) 直接收集 slight/better/much-better 强度，并比较 margin/scaled DPO；
- [Reward Difference Optimization](https://arxiv.org/html/2408.09385) 按 reward gap 重加权 pair；
- [DPO with Rating Information](https://arxiv.org/html/2602.00603) 系统研究 ranking + rating gap，并给出 RDPO/RIPO/ML-RDPO。

因此本文不能把“rubric score gap 进入 DPO”单独作为创新。最合适的定位是：**MMPO 式 margin-to-probability 是 nominal preference construction；新问题是这个 nominal probability 可能因 rubric omission 而系统性失真。**

---

## 4. 推荐的主 \(s_i\)：Rubric-Blind Information Retention

### 4.1 审计数据不能由 rubric 指挥生成

若让生成器逐条读取 criterion 并产生 pass/fail response，再测 score coupling，这已经非常接近：

- [RADAR](https://arxiv.org/html/2608.01810)：criterion-conditioned high/low probes，得到 directional coupling/leakage matrix；
- [Rubrics on Trial](https://arxiv.org/html/2607.15092)：围绕候选 criterion 构造 pass/fail pairs，再用 rubric-blind judge 决定 criterion 是否有用。

更重要的是，criterion-conditioned generator 只能在 rubric 已命名的空间里变化，天然不适合发现 omitted signal。

因此本文的关键设计应是 **rubric-blind generation**：探针生成器只看任务 $x_i$，不看 $\mathcal C_i$，也不被要求沿某个已知 criterion 改写。

### 4.2 审计协议

对每个 instance-specific rubric $\mathcal C_i$：

1. 固定一个生成协议 $\Pi$，从多个模型、温度和 decoding seeds 生成 $M$ 个独立回答对

   $
   (\tilde y_{ij}^{A},\tilde y_{ij}^{B})\sim\Pi(\cdot\mid x_i),
   \qquad j=1,\ldots,M.
   $

   生成器不能看到 $\mathcal C_i$。
2. 一个 rubric-blind holistic panel 只看 $x_i,\tilde y^A,\tilde y^B$，输出

   $
   p^{H}_{ij}=P(\tilde y^A\succ\tilde y^B\mid x_i).
   $

   可由少量人工标注校准，再以冻结、多模型 LLM panel 扩展。panel 不需要给遗漏因素命名。
3. 当前 rubric evaluator 分别给两个 probe response 打分，得到

   $
   m^{\mathcal C}_{ij}
   =r_{\mathcal C_i}(x_i,\tilde y^A_{ij})
   -r_{\mathcal C_i}(x_i,\tilde y^B_{ij}),
   \qquad
   p^{\mathcal C}_{ij}=\sigma(m^{\mathcal C}_{ij}/\tau).
   $

### 4.3 一个简单、有界、显式依赖 rubric 的定义

只保留 holistic panel 有明确偏好的 probes：

$
\mathcal J_i
=\left\{j:\left|p^H_{ij}-\tfrac12\right|\ge\gamma\right\}.
$

把 holistic winner 放在前面后，检查当前 rubric 是否也给它一个方向正确、且不小于阈值的 margin。定义

$
\operatorname{Ret}_{\Pi}(\mathcal C_i;x_i) =
\frac{1}{|\mathcal J_i|}
\sum_{j\in\mathcal J_i}
\mathbf 1\!\left[m^{\mathcal C}_{ij}>\epsilon\right].
$

最终的正向信息权重是

$
\boxed{
s_i
=s_{\min}+(1-s_{\min})
\operatorname{Ret}_{\Pi}(\mathcal C_i;x_i).
}
$

这一定义是 $\mathcal C_i$ 的函数：固定 $x_i,\Pi$、holistic panel、rubric evaluator 和阈值后，改变 rubric 会改变 $m^{\mathcal C}$，进而改变 $s_i$。它不使用当前训练 pair $(y_i^w,y_i^l)$。

解释：

- \(s_i\approx1\)：在 rubric-blind probes 上，凡是整体判断认为重要的差异，rubric 基本都能感知并给出一致方向；
- \(s_i\approx s_{\min}\)：存在大量整体偏好很明确、但 rubric 打成接近平局或反向的 blind spots；
- \(1-\operatorname{Ret}_{\Pi}\) 是相对于审计分布 \(\Pi\) 的不足，不是声称恢复了某个未知维度。

若 \(|\mathcal J_i|\) 太小，说明 audit probes 没制造出足够的可判别差异。此时不能默认 \(s_i=1\)，应增加 probes，或返回区间/全局 prior。

它的反面就是 blind-spot rate：

\[
\operatorname{BSR}_{\Pi}(\mathcal C_i)
=1-\operatorname{Ret}_{\Pi}(\mathcal C_i).
\]

这个主版本不需要 \(H_{\mathcal V}\)、额外预测器或新网络。

### 4.4 可选的连续版本

若不希望结果依赖阈值 \(\gamma,\epsilon\)，可令 holistic decisiveness

\[
\omega_{ij}=2\left|p^H_{ij}-\tfrac12\right|
\]

并用有界 JS discrepancy：

\[
d^{\rm JS}_{\Pi}(\mathcal C_i;x_i)
=
\frac{
\sum_j\omega_{ij}\,
\operatorname{JS}\!\left(
\operatorname{Ber}(p^H_{ij}),
\operatorname{Ber}(p^{\mathcal C}_{ij})
\right)
}{
\log 2\sum_j\omega_{ij}
}.
\]

再令

\[
s_i^{\rm JS}=s_{\min}+(1-s_{\min})(1-d^{\rm JS}_{\Pi}).
\]

建议正文先用 retention/BSR 解释，实验中把连续 JS 版本作为平滑 estimator 与消融。

### 4.5 该指标的准确命名

不要写成无条件的 “true rubric completeness”。建议写：

> \(s_{\Pi}(\mathcal C_i;x_i)\) measures rubric information retention relative to a fixed rubric-blind audit distribution and a calibrated holistic reference panel.

若全程使用 LLM panel，则称 **panel-relative adequacy**；若有人类 audit labels，可进一步验证其 human-relative validity。

---

## 5. 与当前 robust DPO 的最精炼接口

### 5.1 Rubric score 与“赢多少”

先归一化 rubric reward：

\[
r_{\mathcal C_i}(x_i,y)
=
\frac{\sum_{k=1}^{K_i}w_{ik}v_{ik}(x_i,y;c_{ik},a_{ik})}
{\sum_{k=1}^{K_i}w_{ik}}.
\]

训练 pair 上的 rubric margin 为

\[
m_i=r_{\mathcal C_i}(x_i,y_i^w)-r_{\mathcal C_i}(x_i,y_i^l).
\]

按 MMPO/Bradley–Terry 思路构造 nominal preference：

\[
\bar p_i=\sigma(m_i/\tau),
\qquad
P_i^0=\operatorname{Ber}(\bar p_i).
\]

\(m_i\) 只回答“在当前 rubric 下赢多少”，不回答 rubric 是否完整。

### 5.2 \(s_i\) 只进入 ambiguity cost

主方案继续使用

\[
\mathcal U_D
=\left\{
\mathbf Q:
\frac1n\sum_i s_iD(Q_i,P_i^0)\le\rho
\right\}.
\]

- 高 \(s_i\)：rubric 在独立 audit 中保留的信息多，adversary 改动 \(P_i^0\) 的代价高；
- 低 \(s_i\)：rubric blind spots 多，adversary 可更便宜地改变该 preference probability。

这使符号方向最自然，因此推荐正向的 \(s_i\)，而不是把 deficiency 直接放进距离。

### 5.3 不要在主目标中重复使用 \(s_i\)

一种常见替代是把 nominal probability 收缩为

\[
\tilde p_i=\tfrac12+s_i(\bar p_i-\tfrac12).
\]

它可作为 non-robust baseline，但主方法若已经让 \(s_i\) 控制 ambiguity cost，就不建议再同时收缩 \(P_i^0\)，否则同一不确定性会被计算两次。

### 5.4 若 binary winner 来自独立人工而非 rubric

若人类只给方向、rubric 只补充强度，则低 \(s_i\) 不应抹掉人工方向。可以令

\[
\bar p_i=\sigma(\lambda_0+s_im_i/\tau),
\]

其中 \(\lambda_0>0\) 表示独立人类 binary label 的基础证据。本文当前设定更接近 rubric-generated chosen/rejected，因此主版本仍可用 \(\bar p_i=\sigma(m_i/\tau)\)。

---

## 6. 为什么不推荐 Fisher information 作为主 \(s_i\)

Fisher / IRT information 可以衡量：

- 已写入 criterion 对某个 latent trait 区间的区分精度；
- evaluator 参数或 criterion score 的局部估计精度；
- 哪些已观测 items 对当前 response population 最有信息。

但它不能回答：“整个 instrument 是否遗漏了另一个决定性 construct？”一个只精确测量文风、完全遗漏事实性的 rubric，可以有很高 Fisher information，却仍然内容无效。心理测量文献也把 item information、reliability 与 content validity 区分开；后者要求任务/construct domain 与外部证据。[IRT overview](https://pmc.ncbi.nlm.nih.gov/articles/PMC4096146/)；[content-validity process](https://pubmed.ncbi.nlm.nih.gov/11696942/)。

因此 Fisher 适合做辅助 ablation：检测 rubric 对 **已测信号** 的 precision；不适合命名为 unmeasured-information sufficiency。

---

## 7. Bayesian 是否意味着再训练一个网络

不一定。Bayesian 只是把未知量写成概率分布。例如对每个 rubric 的 audit outcomes，可以使用：

- beta-binomial / Dirichlet-multinomial 模型估计 blind-spot rate；
- hierarchical Bayesian logistic calibration 估计 \(p^H,p^{\mathcal C}\)；
- bootstrap 或 Bayesian posterior 给 \(s_i\) credible interval。

这些都可以是小型统计模型，无需训练深度网络。若使用一个神经 evaluator 来产生 scores，那是因为文本评分本身需要模型，不是因为 Bayesian 必然等于网络。

例如令 \(h_i\) 为通过 audit 的 probe 数，\(M_i=|\mathcal J_i|\)，可直接使用

\[
u_i\mid h_i,M_i\sim\operatorname{Beta}(h_i+1,M_i-h_i+1),
\]

并取保守下置信界

\[
s_i^{\rm LCB}
=s_{\min}+(1-s_{\min})Q_{\delta}(u_i),
\]

其中 \(Q_{\delta}\) 是 posterior 的低分位数。probe 很少时，\(s_i^{\rm LCB}\) 会自动更保守。

但 Bayesian posterior 仍只来自它看到的 evidence。若 labels 完全由当前 rubric 自己生成，Bayesian 也不能凭空识别 rubric 的遗漏。它更适合作为 RBIR 的 uncertainty estimator，而不是 \(s_i\) 的定义。

---

## 8. 完全文本式 \(s(\mathcal C)\) 能做什么

若完全禁止任何 audit responses，只能得到 structural prior，而不能得到 preference sufficiency。可参考：

- [RIFT](https://arxiv.org/html/2604.01375)：subjective、non-atomic、ungrounded、misaligned/rigid、missing criteria、hackable、low signal、redundant criteria；
- [PReMISE](https://arxiv.org/html/2605.30803)：atomicity、internal consistency、response observability、operationalizability、unambiguous scope；
- criterion embedding / entailment Gram matrix 的 log-det 或 effective rank：只测语义非冗余度。

一个简单 baseline 是

\[
s_{\rm text}(\mathcal C_i;x_i)
=1-\sum_{f\in\mathcal F}\alpha_f\,
\widehat P(f\mid x_i,\mathcal C_i),
\qquad \sum_f\alpha_f=1,
\]

其中 \(\mathcal F\) 是 RIFT failure types。它可用于便宜的 pre-screen，但不能作为主充分性结论：写得清楚、互不重复的 rubric 仍可能漏掉重要偏好因素。

---

## 9. 与最新近邻工作的边界

| 近邻 | 已做什么 | 本文必须保留的差异 |
|---|---|---|
| [PReMISE](https://arxiv.org/html/2605.30803) | 用自然 preference data 审计 structural adequacy、reliability、preference fit、robustness；报告 judge accuracy 和 effective dimension | 不再声称“第一次评估 rubric quality”；聚焦 rubric-blind audit 得到 instance-level trust scalar，并连接 unmeasured-confounding DRO |
| [RIFT](https://arxiv.org/html/2604.01375) | rubric failure taxonomy 与自动诊断 | RBIR 不枚举具体遗漏 criterion，而在 response space 直接度量整体偏好被 rubric 忽略的程度 |
| [RADAR](https://arxiv.org/html/2608.01810) | criterion-conditioned probes 与 coupling matrix | 生成器必须不看 rubric；目标是 omitted blind spots，不是已写 criteria 的依赖/冗余 |
| [Rubrics on Trial](https://arxiv.org/html/2607.15092) | 为候选 criterion 构造 pass/fail pair，用 blind judge 筛选并扩展 rubric | 本文不生成/补充 criterion；只审计整个固定 rubric 对 rubric-blind response variation 的 retention |
| [MMPO](https://aclanthology.org/2024.findings-emnlp.792/) / [rating-DPO](https://arxiv.org/html/2602.00603) | 使用已知 score gap/strength | 本文的新增量不是 score gap，而是 score gap 在 rubric omission 下应被信任多少 |
| [Visual rDPO](https://arxiv.org/html/2604.13029) / [RLCF](https://arxiv.org/html/2507.18624) | rubric scoring 与 pair mining | 本文保留连续 nominal probability，并显式处理 rubric 未测信息导致的分布不确定性 |

鉴于 2026 年 rubric-auditing 文献已很拥挤，最安全的论文 claim 是：

> We estimate an instance-specific, rubric-level information-retention weight from rubric-blind response-space probes, without naming or completing omitted criteria, and use it as the transportation cost of preference-distribution perturbations in robust preference optimization.

不要只 claim “rubric adequacy metric”，也不要只 claim “robust DPO”。

---

## 10. 最小可执行实验

### 10.1 审计指标是否真的响应 omission

从较完整 rubrics 出发，按不同强度随机 mask criteria，或者隐藏由专家判为重要的一部分，但评估时不向 RBIR 暴露 mask 语义。期望：mask 越强，\(s_i\) 越低、BSR 越高。

### 10.2 区分 pair margin 与 rubric adequacy

做二维分桶：

- 大 \(m_i\)、高 \(s_i\)：强且可信的 preference；
- 大 \(m_i\)、低 \(s_i\)：rubric 很自信但 audit 显示 blind spots；
- 小 \(m_i\)、高 \(s_i\)：完整 rubric 认为两答确实接近；
- 小 \(m_i\)、低 \(s_i\)：既接近又不可靠。

这能直接证明 \(m_i\neq s_i\)。

### 10.3 关键 baselines

- uniform \(s_i=1\)；
- rubric length / criterion count；
- RIFT/PReMISE structural score；
- Fisher/score variance；
- criterion-conditioned RADAR-style self-effect/effective dimension；
- natural-pair PVI/RPI（只作 pair-level validation）；
- RBIR without rubric-blindness；
- RBIR with single vs multi-generator、single vs multi-judge；
- non-robust MMPO 与 uniform robust learner。

### 10.4 防止同源 LLM 产生虚假一致

- probe generator、holistic panel、rubric judge 使用不同 model families；
- A/B order swap 与 rubric paraphrase；
- held-out human audit subset；
- cross-fitting calibration temperature \(\tau\)；
- 报告 \(s_i\) 的 bootstrap/Bayesian interval；
- 对相同 rubric 的 \(s_i\) 做 test-retest reliability。

### 10.5 \(s\) 与 \(\rho\) 的尺度

加权 ambiguity set 中 \(s\) 的整体尺度与 \(\rho\) 可互相替代。实验中应固定均值，例如

\[
\tilde s_i=\frac{s_i}{n^{-1}\sum_j s_j},
\]

并在 validation set 上选择 \(\rho\)，否则改动 \(s\) 的平均值可能被误认为个性化 weighting 的收益。

---

## 11. 最终推荐的一句话方法链

\[
\boxed{
\text{rubric score margin }m_i
\longrightarrow
P_i^0=\operatorname{Ber}(\sigma(m_i/\tau))
\quad;
\quad
\text{rubric-blind audit }\longrightarrow s_i
\quad;
\quad
\sum_i s_iD(Q_i,P_i^0)\le n\rho.
}
\]

- \(m_i\)：在当前 rubric 下赢多少；
- \(s_i\)：该 rubric 在独立、rubric-blind response variation 上保留了多少整体 preference information；
- \(Q_i\)：未测信息可能诱导的真实 preference distribution；
- minimax：处理该剩余不确定性的下游学习器。

这比旧的 \(R_i/H_{\mathcal V}/A\) 表达更简洁，也真正把 rubric-level quality、pair-level strength 与 robust learning 三者分开。
