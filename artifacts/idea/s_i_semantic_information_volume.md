# 无人工标签的 rubric-level weight：Semantic Information Volume

> **状态：历史调研材料（非执行规范）。** 本文件保留当时的候选方案与术语，可能与最终设计不同。正式实验只以根目录 [`PLAN.md`](../../PLAN.md)、[`CHECKLIST.md`](../../CHECKLIST.md) 和 [`final_experiment_matrix_audit.md`](../scout/final_experiment_matrix_audit.md) 为准；尤其不得从本文沿用旧称 “Scaled DPO”、旧 scorer/gate 或旧数据映射。

> **状态：已降级为消融方案。** 进一步文献核对后，本文不再把 text-embedding volume 作为主 $s_i$。它只能度量 rubric 文本的语义多样性，不能证明 judge 实际获得了独立测量信息。当前主方案是 score-covariance-based Rubric Operational Information Volume（ROIV），完整定义和增量实验设计见 [Unmeasured Rubrics_ICLR27.md](../../Unmeasured%20Rubrics_ICLR27.md)。

## 1. 本轮约束与结论

当前要求是：

- instance-specific rubric \(\mathcal C_i\) 在制作时已经吸收任务信息；
- 权重应写成 \(s_i=s(\mathcal C_i)\)；
- 不使用当前 DPO pair \((y_i^w,y_i^l)\)；
- 不需要人工 preference label；
- 指标应尽量简单，并能在明确模型假设下证明“越高表示 rubric 表达的信息越多”。

据此，旧的 Rubric-Blind Information Retention 不再作为主方案。本文推荐 **Rubric Semantic Information Volume（RSIV）**：将 rubric criteria 视为潜在质量空间中的测量方向，用一个冻结文本 encoder 得到方向向量，再以 Fisher information matrix 的 log-determinant 衡量 rubric 张成的非冗余信息体积。

主方法只需要：一次 criterion embedding 和一次小矩阵 log-determinant；不生成回答、不训练额外网络、不调用人工标签。

## 2. 最终定义

Rubric 为

$
\mathcal C_i=\{(c_{ik},a_{ik},w_{ik})\}_{k=1}^{K_i}.
$

用固定、冻结的 text encoder $\phi$ 编码 criterion 及其 reference / guidance：

$$
e_{ik} =
\frac{\phi(c_{ik}\oplus a_{ik})}
{\|\phi(c_{ik}\oplus a_{ik})\|_2},
\qquad
\bar w_{ik}=\frac{w_{ik}}{\sum_l w_{il}}.
$$

令 $E_i\in\mathbb R^{K_i\times d}$ 的第 $k$ 行为 $e_{ik}^{\top}$，并令

$$
W_i=\operatorname{diag}(\bar w_{i1},\ldots,\bar w_{iK_i}).
$$

定义 semantic information matrix：

$$
F(\mathcal C_i)=E_i^{\top}W_iE_i
=\sum_k\bar w_{ik}e_{ik}e_{ik}^{\top}.
$$

定义 raw information score：

$$
\mathcal I(\mathcal C_i) = \frac12\log\det\!\left(I_d+\lambda F(\mathcal C_i)\right).
$$

实际实现不需要计算 $d\times d$ determinant。由 Sylvester determinant identity：

$$
\mathcal I(\mathcal C_i) = \frac12\log\det\!\left(
I_{K_i}+\lambda W_i^{1/2}E_iE_i^{\top}W_i^{1/2}
\right).
$$

最终权重为：

$$
\boxed{
s_i=s(\mathcal C_i)
=s_{\min}+(1-s_{\min})
\left[1-e^{-\mathcal I(\mathcal C_i)}\right].
}
$$

若需要反向的“未测部分”指标，先定义 raw deficiency：

$$
\tilde u_i=e^{-\mathcal I(\mathcal C_i)},
$$

进入算法的有界 deficiency 为：

$$
u_i=1-s_i=(1-s_{\min})\tilde u_i.
$$

所有 rubrics 共用同一个 encoder、$\lambda$ 和 $s_{\min}$。embedding 归一化后可将主实验固定为 $\lambda=1$，其他值只做敏感性分析。因为后续 ambiguity radius $\rho$ 与 weight 的全局尺度存在等价性，$\lambda$ 不应针对单条 rubric 调节。

## 3. 为什么 log-det 比简单计数更合适

设 $\mu_1,\ldots,\mu_r$ 是 $F(\mathcal C)$ 的非零 eigenvalues，则

$
\mathcal I(\mathcal C)
=\frac12\sum_{j=1}^{r}\log(1+\lambda\mu_j).
$

因此：

1. **非冗余信息增加。** 新 criterion 若提供新的语义方向，会增加新的 eigenvalue；
2. **重复 criterion 不会被简单计数。** 若 criteria 语义近似共线，信息集中在原有 eigen-direction；在权重归一化后，完全重复且仅重新分配权重的 criteria 不改变 \(F\)；
3. **重要性进入指标。** 低权重 criterion 对 \(F\) 的贡献小；
4. **存在自然的 diminishing return。** 每个方向的边际增益为 \(\log(1+\lambda\mu)\)，不会随 criterion 数量线性无界增长。

Criterion count \(K\) 会把重复标准全部算作新信息；平均 pairwise cosine 只看冗余比例，不衡量总体信息量；effective rank 只看 eigenvalue 的相对均匀程度，即使所有 eigenvalues 都极小也可得到很高 effective rank。Log-det 同时保留“有多少方向”和“这些方向贡献了多少信息”。

## 4. 理论解释：Fisher、mutual information 与未测不确定性

作如下明确的 latent semantic measurement assumption：

$$
h\sim\mathcal N(0,I_d),
$$

其中 $h$ 表示 task-relevant response-quality factors。Rubric criterion $k$ 对应测量方向 $e_k$：

$$
z_k=e_k^{\top}h+\varepsilon_k,
\qquad
\varepsilon_k\sim
\mathcal N\!\left(0,(\lambda\bar w_k)^{-1}\right),
$$

并假设不同 criterion noise 条件独立。则关于 \(h\) 的 Fisher information matrix 为：

$$
J_{\mathcal C}=\lambda E^{\top}WE=\lambda F(\mathcal C).
$$

在线性 Gaussian 模型中：

$$
I(h;z\mid\mathcal C)
=
\frac12\log\det(I_d+J_{\mathcal C})
=\mathcal I(\mathcal C),
$$

且

$$
\Sigma_{h\mid z,\mathcal C}
=(I_d+J_{\mathcal C})^{-1}.
$$

于是：

$$
H(h\mid z,\mathcal C)
=H(h)-\mathcal I(\mathcal C).
$$

进一步，posterior confidence ellipsoid 与 prior confidence ellipsoid 的 volume ratio 为：

$$
\frac{\operatorname{Vol}_{\rm post}}
{\operatorname{Vol}_{\rm prior}}
=
\sqrt{\frac{\det\Sigma_{h\mid z,\mathcal C}}
{\det I_d}}
=e^{-\mathcal I(\mathcal C)}.
$$

因此 $1-e^{-\mathcal I(\mathcal C)}$ 可解释为 rubric 对 latent uncertainty volume 的相对削减量。这正是推荐权重的来源，而不是任意的 heuristic normalization。

严格的理论结论是：

> 在 criterion embeddings 等价于 task-relevant latent measurement directions 的线性 Gaussian 模型下，$\mathcal I(\mathcal C_1)>\mathcal I(\mathcal C_2)$ 当且仅当 $\mathcal C_1$ 相对于 $\mathcal C_2$ 带来更大的总 expected information gain，并对应更小的 posterior uncertainty volume。

这里比较的是总体 uncertainty volume。Scalar log-det ordering 不保证每一个预先指定的 preference direction 都有更小 posterior variance；若要这种最坏方向保证，需要比较 Fisher matrices 的 Loewner order，或使用 minimum eigenvalue（E-optimality）。但在“不知道遗漏方向是哪一个”的设定下，D-optimal volume 比指定某个方向更自然。

## 5. 与 unmeasured confounding 的关系与边界

若真实 preference-relevant latent space 在不同 rubrics 之间固定，则更大的 $\mathcal I(\mathcal C)$ 表示 rubric 覆盖的非冗余 measurement volume 更大，留在 posterior 中的 latent uncertainty volume更小。本文据此将

$$
\tilde u(\mathcal C)=e^{-\mathcal I(\mathcal C)}
$$

作为 unmeasured-information proxy，并让其通过 $s(\mathcal C)$ 控制 robust ambiguity cost。

但以下不可识别性必须写清：如果一个 rubric 加入彼此不同、却与真实 preference 完全无关的 criteria，text-only log-det 仍可能增加。仅凭 rubric 文本无法区分：

$$
U_1(y)=r_{\mathcal C}(y)
\quad\text{与}\quad
U_2(y)=r_{\mathcal C}(y)+U_{\rm miss}(y).
$$

两个世界中的 $\mathcal C$ 完全相同，任何纯 $f(\mathcal C)$ 也相同。因此不能在无结构假设、无外部 evidence 时声称估计了 absolute human-preference completeness。

本方法的合理性建立在用户已经给定的建模前提上：instance-specific rubric 在制作时已经针对任务，主要剩余问题是 criteria 数量不足、语义重叠或覆盖方向不足。论文中应称其为：

> rubric-expressed semantic information volume / semantic measurement sufficiency proxy

而不是：

> identifiable true fraction of all human preference information.

## 6. 候选指标比较

| 指标 | 是否纯 \(s(\mathcal C)\) | 无人工标签 | 主要问题 | 结论 |
|---|---:|---:|---|---|
| Criterion count | 是 | 是 | 重复 criterion 也增加 | 过弱 baseline |
| 平均 cosine distance | 是 | 是 | 只看平均冗余，不衡量总体 volume | baseline |
| Embedding effective rank | 是 | 是 | 只看 eigenvalue 比例，不看信息强度 | 辅助诊断 |
| **Semantic Fisher log-det** | **是** | **是** | 依赖 embedding-as-measurement 假设 | **主方法** |
| Score entropy | 否，需要 response scores | 是 | 高随机噪声也可产生高 entropy | 不推荐 |
| Repeated-score MI / ICC | 否，需要无标签 responses 与重复评分 | 是 | 测 reliability / channel capacity，不测内容覆盖 | 可选 validation |
| IRT/Fisher over quality level | 否，需要 latent quality level 或伪标签 | 可自动合成 | 对遗漏 construct 不敏感，且容易 self-fulfilling | 不作主方法 |
| PVI / no-rubric comparison | 否，需要回答对和外部 target | 可用 AI label | 衡量 preference prediction，而非纯 rubric text | 外部 validation |

## 7. 文献定位

1. [From Implicit Weights to Explicit Rubrics / Auto-Rubric](https://arxiv.org/html/2510.17314) 已使用 rubric embedding 的 coding-rate log-det 来鼓励 semantic coverage 与非冗余性。本文不能将 log-det 本身声称为创新；新增价值只能是把 task-specific rubric information volume 映射成 unmeasured-confounding-aware robust-DPO weight。
2. [PReMISE](https://arxiv.org/html/2605.30803) 从 criterion score covariance 的 eigen-spectrum 计算 effective dimensionality，并明确区分 structural adequacy、reliability、preference fit 与 robustness。它支持“非冗余维数是 rubric anatomy 的一个轴”，同时说明该轴不等于完整 validity。
3. [Maximal Coding Rate Reduction](https://proceedings.neurips.cc/paper/2020/hash/6ad4174eba19ecb5fed17411a34ff5e6-Abstract.html) 给出了 log-det coding rate 的信息论与几何依据。
4. [A Brief Note on the Bayesian D-Optimality Criterion](https://arxiv.org/abs/2212.11466) 推导了在线性 Gaussian inverse problem 中，maximizing expected information gain 等价于 minimizing posterior covariance log-determinant。
5. [Near-Optimal Sensor Placements in Gaussian Processes](https://www.jmlr.org/papers/v9/krause08a.html) 将 mutual information 与 D-optimal sensor placement 联系起来，并讨论信息增益的 submodularity / diminishing returns。
6. [The Effective Rank](https://core.ac.uk/download/pdf/147929764.pdf) 用 eigenvalue entropy 定义有效维数，适合作为 redundancy diagnostic，但不是总 information volume。
7. [PReMISE](https://arxiv.org/html/2605.30803) 和 [RIFT](https://arxiv.org/abs/2604.01375) 均表明 rubric quality 包含多个不同轴；任何单一 intrinsic metric 都不应被包装为所有意义上的 rubric quality。
8. [Messick's validity framework](https://www.ets.org/research/policy_research_reports/publications/report/1994/hxpp.html) 强调 measurement validity 是对 score meaning / use 的证据论证，而不是 measurement text 的一个无条件属性。

## 8. 最小实现

对每个 $\mathcal C_i$：

1. 使用同一个 frozen sentence encoder 得到 $e_{ik}$；
2. 对 embedding 做 $\ell_2$ normalization；
3. 用 rubric weights 构造 $W_i$；
4. 计算 $G_i=W_i^{1/2}E_iE_i^{\top}W_i^{1/2}$；
5. 计算 $\mathcal I_i=\tfrac12\operatorname{slogdet}(I_{K_i}+\lambda G_i)$；
6. 计算 $s_i=s_{\min}+(1-s_{\min})(1-e^{-\mathcal I_i})$；
7. 在整个训练阶段固定 $s_i$。

建议默认：所有 criteria 权重为正，$\lambda$ 对整个数据集固定；若原始 rubric 没有 importance weights，则取 $w_{ik}=1$。

## 9. 最小自动验证，不是主指标的一部分

无需人工标签即可验证该 proxy 的内部合理性：

1. **重复 criterion 测试：** 将一个 criterion 原样复制并均分原权重，$s(\mathcal C)$ 应基本不变；
2. **criterion masking：** 随机删除 criterion，平均 $s(\mathcal C)$ 应下降；
3. **非冗余添加：** 添加与已有 criteria embedding 近似正交、且任务相关的 criterion，$s(\mathcal C)$ 应上升；
4. **paraphrase stability：** criterion 等价改写后，$s(\mathcal C)$ 应稳定；
5. **encoder sensitivity：** 使用两个冻结 encoder 计算 $s$ 排序的 Spearman correlation。

这些测试只验证指标按定义工作，不验证 absolute human-preference completeness。

## 10. 推荐的论文 claim

可安全写成：

> We quantify the task-conditioned semantic information expressed by an instance-specific rubric using the D-optimal volume of its weighted criterion-embedding information matrix. Under a latent Gaussian measurement model, the resulting score equals expected information gain and monotonically decreases posterior uncertainty volume. We use this label-free rubric functional as the transportation cost in robust preference optimization.

不要写成：

> Our statistic identifies the true amount of all omitted human-preference information from rubric text alone.
