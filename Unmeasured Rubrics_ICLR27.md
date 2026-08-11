# Unmeasured Rubrics\_ICLR27

目前的 rubric-based DPO 依赖 rubrics 构造偏好数据。即使 rubric 包含多个 criteria，它仍可能遗漏影响潜在偏好的因素，使 rubric-induced preference label 发生系统性 misspecification。本文不把 robust learning 本身作为创新，而是研究：能否为每个实际 rubric criterion set $\mathcal C_{gi}$ 得到一个无人工标注的 sufficiency proxy $s_{gi}=s(\mathcal C_{gi})$，再用它校准对应 preference ambiguity。相同 criterion set 必须共享同一个 $s$；若全数据使用同一 rubric，就退化为常数 $s_g$；若 producer 为每个 prompt 生成不同 rubric，则逐 rubric 计算不同的 $s_{gi}$。

若论文使用 “unmeasured confounding” 这一因果术语，需要额外给出 treatment、outcome、latent confounder 和因果图；否则更准确的表述是 **unmeasured preference factors**、**rubric incompleteness** 或 **rubric-induced preference misspecification**。本文后续使用后一组表述。



# Notation

令 $g\in\{1,\ldots,G\}$ 索引 rubric 系统（或使用该 rubric 的数据集/实验条件），给定
$$
\mathcal D_g=\{z_{gi}\}_{i=1}^{n_g},
\qquad
z_{gi}=(x_{gi},y_{gi}^w,y_{gi}^l).
$$
producer/system $g$ 可以给不同样本分配不同 rubrics。记 $\mathcal C_{gi}$ 为样本 $i$ 实际采用的 criterion set；prompt-specific generator 对应 $\mathcal C_{gi}=G_g(x_{gi})$，但 $x_{gi}$ 与 generator $G_g$ 都不属于 rubric 本体。对 canonical criterion JSON 做 exact hash 得到 `rubric_id`；若 $\mathcal C_{gi}=\mathcal C_{gj}$，两条样本必须共享缓存的 $s(\mathcal C)$。



$x_{gi}$：rubric 系统 $g$ 下的第 $i$ 个 prompt

$y_{gi}^w$：observed chosen response

$y_{gi}^l$：observed rejected response

$\mathcal C_{gi}$：样本 $i$ 实际用于评分的 rubric criterion set，包括 criterion wording 及其绑定的 reference/guidance、score anchors、方向和 importance。prompt、response、producer 名、generator 与共享 calibration evaluator均不属于 $\mathcal C_{gi}$。本文使用 $s_{gi}=s(\mathcal C_{gi})$；它可以随样本变化的唯一原因是 rubric 发生变化，而不是把 $x$、chosen/rejected、margin 或 loss 直接输入一个 sample-weight function。

$K_{gi}$：rubric $\mathcal C_{gi}$ 中 criterion 的数量

对任一 rubric，有 $\mathcal C_{gi} = \left\{ (c_{gik},a_{gik},w_{gik}) \right\}_{k=1}^{K_{gi}}$，其中：

$c_{gik}$：第 $k$ 个评价标准；

$a_{gik}$：该标准对应的 reference 或 evaluation guidance；

$w_{gik}$：该标准的重要性权重。

另定义冻结的 **margin-producing functional** $r_g^{m}$，它与共享 calibration instrument $v_\psi^{\rm cal}$ 不是同一个对象。对 UltraFeedback、WildChecklists 与 HelpSteer2，$r_g^{m}$ 分别来自发布的 aspect annotations、发布的 aggregate scores和预注册的人类 rating scalarization；对 automatic-rubric controlled track，$r_g^{m}$ 才由 shared RubricARROW criterion scores 按 $\mathcal C_g$ 聚合。若 system 提供逐 criterion scores，则

$$
r_g^{m}(x,y,\mathcal C_{gi})=
\frac{\sum_{k=1}^{K_{gi}}w_{gik}v_{gik}^{m}(x,y,c_{gik},a_{gik})}
{\sum_{k=1}^{K_{gi}}w_{gik}};
$$

若只发布 aggregate score，则直接把该冻结分数作为 $r_g^m$。raw observed rubric margin 统一写为 $m^{\rm raw}_{gi}=r_g^m(x_{gi},y_{gi}^w,\mathcal C_{gi})-r_g^m(x_{gi},y_{gi}^l,\mathcal C_{gi})$。margin 随 response pair 变化，表示在当前 rubric 下“赢了多少”；$s_{gi}$ 只由 rubric identity 决定。共享 calibration scorer 只估计 $s(\mathcal C)$，不会重写 native $m^{\rm raw}$。

若发布的聚合公式含负系数，先把对应的有界 criterion score 反向为“越高越好”，再使用系数绝对值。例如归一化 verbosity 使用 $v^-=1-v$ 与正权重；这与原负系数聚合只相差与 response 无关的常数，因此不改变 pair direction 或 margin。ROIV 的 $W^{1/2}$ 始终只接收非负 importance。

为使不同 rubric system 的 gap 尺度可执行且不泄漏 validation/test，每个 system 只在 training split 的所有 $m^{\rm raw}_{gi}>0$ 上，用 NumPy `quantile(method="linear")` 冻结 $q^{\rm train}_{0.95,g}$，并定义

$$
m_{gi}=\operatorname{clip}\!\left(\frac{m^{\rm raw}_{gi}}{\max(q^{\rm train}_{0.95,g},10^{-8})},0,1\right).
$$

raw exact ties $m^{\rm raw}_{gi}\le0$ 直接丢弃；归一化后 $m_{gi}<0.05$ 作为预注册 near-tie 丢弃。validation/test 只应用 train 的 $q_{0.95,g}$；每个 system 必须满足 $q_{0.95,g}>0$，并保存 tie、near-tie 与 clipping counts。参考 MMPO / Bradley--Terry 的做法，再将归一化 rubric margin 映射为 nominal soft preference：

$$
\bar p_{gi}=
\operatorname{clip}\!\left(\sigma(\gamma m_{gi}),\varepsilon,1-\varepsilon\right),
\qquad
P_{gi}^0=\operatorname{Bernoulli}(\bar p_{gi}),
$$

其中 $\gamma$ 是所有样本共享的 margin temperature，$\varepsilon$ 只用于数值稳定；两者不需要新的人工标注。$m_{gi}$ 是 **pair-level preference strength**：它回答“在当前 rubric 下赢多少”，不回答当前 rubric 是否遗漏了重要信息。

主配置冻结 $\varepsilon=10^{-6}$，并在数据 manifest 中记录触发 clipping 的样本数。nominal、uniform 和 ours 始终共享同一个 $P^0$。若没有样本触发 clipping，$\rho=0$ 与原 MMPO 逐点相同；否则只能称为 **stabilized/clipped MMPO baseline**，不能把数值稳定化后的 objective 冒充未裁剪原公式。

## rubric\-DPO loss

定义 DPO implicit reward：$ \widehat r_\theta(x,y) = \beta \log \frac{ \pi_\theta(y\mid x) }{ \pi_{\mathrm{ref}}(y\mid x) }.$

对应的 policy reward margin 为：$\Delta_\theta(z_{gi})=\widehat r_\theta(x_{gi},y_{gi}^w)-\widehat r_\theta(x_{gi},y_{gi}^l)$。

标准 DPO loss 为：$\ell_{gi}^{\mathrm{DPO},+}(\theta)=-\log\sigma(\Delta_\theta(z_{gi}))$。

需要注意：标准 DPO 的监督接口只有 chosen/rejected。$\Delta_\theta$ 是 policy 自己学出的 implicit margin，不是 rubric 提供的外生“赢多少”。现有 rubric-DPO 通常用 rubric 分数构造或筛选 preference pairs，随后仍训练标准二元 DPO；本文通过 $P_{gi}^0=\operatorname{Bernoulli}(\bar p_{gi})$ 保留 rubric score gap。







# Rubric-level weight——Rubric Operational Information Volume

## 0. 先说明一个不可识别边界

若指标被严格限制为 rubric 文本的确定函数 $f(\mathcal C)$，又不给定任何 reference population、judge behavior 或 preference observations，则“真实还遗漏多少”不可识别。证明很直接：可以构造两个拥有完全相同 rubric 文本的世界；世界一的真实偏好完全由该 rubric 决定，世界二的真实偏好主要由未写入 rubric 的潜在因素决定。两者的 $f(\mathcal C)$ 必然相同，但真实 completeness 不同。

因此，text-only embedding 最多识别 **rubric 内部的语义广度与非冗余性**。若要让 $s$ 更接近 measurement sufficiency，必须明确固定一个不依赖当前 DPO pair 的 calibration environment。本文采用冻结的 probe policy 与一套所有 rubric systems 共享的 reference-free pointwise criterion grader；native judge 与需要 reference 的 grader 只做敏感性分析。在这些环境量固定后，最终仍把统计量记作 rubric-level functional $s(\mathcal C)$。

## 1. 先固定 $s(\mathcal C)$ 的含义

Rubric 的统计信息量不是文字本身的无条件属性；它还依赖于固定的 evaluator 和被测 response population。本文先固定一个 probe policy $\Pi_0$ 和全系统共享的 calibration evaluator $v_\psi^{\rm cal}$，并定义

$$
s_{gi}=s_{\Pi_0,\psi}(\mathcal C_{gi})\equiv s(\mathcal C_{gi}).
$$

这里省略的 $\Pi_0,\psi$ 对所有 rubrics 完全相同。对 rubric criterion JSON 做 exact canonical hash；相同 rubric 只产生一个预先计算并缓存的 scalar，不同 rubric 分别计算。Probe prompts/responses 只用于 Monte Carlo 估计；chosen/rejected、pair margin 和训练 loss 均不进入 $s$。

## 2. 用 rubric 的实际评分行为构造 information matrix

令 $r$ 索引不同的 exact rubrics，$A_r$ 表示冻结 corpus 中实际被分配 rubric $\mathcal C_r$ 的 prompts。全局 rubric 的 $A_r$ 是共同 calibration bank；prompt-specific rubric 通常只有其来源 prompt。如果同一 criterion set 在多个 prompts 上重复出现，则合并这些 prompts 并共享一个 $s_r$。prompt 是应用/测量上下文，不进入 rubric hash。对每个 $x_\ell\in A_r$ 固定采样 $M=16$ 个 responses：

$$
 x_{\ell}\in A_r,
\qquad
\widetilde y_{\ell j}\sim\Pi_0(\cdot\mid x_{\ell}),
\quad j=1,\ldots,M.
$$

使用全系统共享的冻结 `OpenRubrics/RubricARROW-8B-Judge`、官方 JSON criterion prompt、官方 completion-logprob extractor 与同一 parser contract，得到 prompt 内的归一化 criterion-score matrix。冻结 `top_logprobs=10`；在每个 `criteria_met_i` 位置读取 literal ` true`/` false` 的 log-prob，令 $p_i^T=\exp(l_{\rm true})$、$p_i^F=\exp(l_{\rm false})$，并将官方 probability score $p_i^T-p_i^F$ 仿射为 $Z_i=\operatorname{clip}[(1+p_i^T-p_i^F)/2,0,1]$。需要二值判断时唯一使用 $1[p_i^T\ge p_i^F]$。JSON/索引或任一概率缺失时，该 criterion 记 $Z_i=0$ 并计入失败率，不重新调用模型。producer-level parse success 低于 99% 时该 producer 移出主表；单个 rubric failure 与过滤写入 manifest。Prometheus-7B-v2 只在固定 200-prompt subset 做 scorer sensitivity，不进入主 pair、$m$ 或 $s$。

$$
Z_{r\ell}[j,k]
=v_\psi^{\rm cal}(x_{\ell},\widetilde y_{\ell j},c_{rk},a_{rk})\in[0,1],
\qquad
Z_{r\ell}\in\mathbb R^{M\times K_r}.
$$

所有 criterion scores 先统一为“越高越好”。若原聚合存在负系数，则反向该维 score 并使用系数绝对值；例如 HelpSteer2 使用 $v^-=1-v/4$ 与正 importance $0.40$，其 pair gap 与原来的 $-0.40v$ 完全等价。移除 importance 为 0 的 criteria 后，归一化非负 weights，形成 $W_r$。**固定 rubric 和 prompt-specific rubric 都只使用 prompt 内 response covariance**：

$$
\widehat\Sigma_{r\ell}
=\operatorname{Cov}_{j=1}^{M}\!\left(Z_{r\ell}[j,:]\right),
\qquad
G_{r\ell}=W_r^{1/2}\widehat\Sigma_{r\ell}W_r^{1/2}.
$$

定义 **Rubric Operational Information Volume (ROIV)**：

$$
\boxed{
\mathcal I_{\rm op}(\mathcal C_r)
=\frac1{|A_r|}\sum_{\ell\in A_r}\frac12\log\det\!\left(I_{K_r}+\lambda G_{r\ell}\right),
\qquad
s_r=\operatorname{clip}\!\left(
1-\exp\{-\mathcal I_{\rm op}(\mathcal C_r)\},
\epsilon_s,1
\right).
}
$$

其中 $\lambda>0$ 是所有 rubrics 共享的 signal-to-noise scale。经过 $s=1-e^{-\mathcal I}$ 的非线性映射后，改变 $\lambda$ 一般会改变 rubrics 间的相对 $s$，不能被一个固定 $\rho$ 完全吸收。因此主实验对所有 rubrics 预先固定同一个 $\lambda=1$，并做全局 $\lambda\in\{0.1,1,10\}$ 敏感性，不对单个 rubric 调参。

若 baseline 本来就重复调用 judge，可选地在每个 prompt 内用 within-response covariance 估计评分噪声，并用 signal covariance 替换上式中的 $\widehat\Sigma_{r\ell}$。全局 rubric 先算每个 prompt 的 scalar 再在 $A_r$ 内平均；prompt-specific unique rubric 通常由一个 prompt 的 response variation 估计。后者严格说是 rubric 在其冻结 assignment context 下的 operational statistic，记号中省略 context，但 prompt 仍不属于 rubric 本体。

$$
\widehat\Sigma_{r\ell,\rm sig}
=\left[
\operatorname{Cov}_{j}(\bar Z_{r\ell}[j,:])
-\frac1R\widehat\Sigma_{r\ell,\rm within}
\right]_+,
$$

主方案可直接使用 baseline 已聚合的 scores；noise subtraction 只作为可靠性消融，不增加新的人工监督。

## 3. 理论解释

对固定 prompt $x$，假设 probe response 的潜在 task-relevant quality 为 $h\mid x\sim\mathcal N(0,\Sigma_{h\mid x})$，rubric-conditioned score vector 满足

$$
z=A_{\mathcal C}h+\epsilon,
\qquad
\epsilon\sim\mathcal N(0,\lambda^{-1}W^{-1}).
$$

则 $A_{\mathcal C}\Sigma_{h\mid x}A_{\mathcal C}^{\top}$ 是同一 prompt 下不同 responses 引起的 score-signal covariance，且

$$
I(h;z\mid x,\mathcal C)
=\frac12\log\det\!\left(
I+\lambda W^{1/2}A_{\mathcal C}\Sigma_{h\mid x}A_{\mathcal C}^{\top}W^{1/2}
\right).
$$

因此在固定 probe policy、rubric-assignment population 和 judge 的模型下，prompt 内 mutual information 再在同一 exact rubric 的 assignment set 内平均得到 $\mathcal I_{\rm op}$；它越大，rubric 实际输出的稳定、非冗余 measurement information 越多。该结论不要求找出遗漏的是哪个具体维度，只度量当前 rubric channel 已经测到了多少独立信息。

这个量仍不是无假设下可识别的“全部真实人类偏好覆盖率”。它应称为 **operational measurement-information proxy**；真实未测偏好因素仍由后续 sensitivity/minimax analysis 处理。

## 4. 只有 helpfulness / consistency 时会怎样

若 rubric 只有 `helpfulness` 与 `consistency`：

- 仅对两个词做 embedding，只能说明词义不同，不能说明 judge 真正执行了两个不同测量；
- 在 ROIV 中，若两列分数几乎相同，$\widehat\Sigma_g$ 近似 rank-one，第二个 criterion 几乎不增加 information volume；
- 若两列都近似常数，$s(\mathcal C)$ 接近 0；
- 若所有样本使用同一套 `helpfulness + consistency` criteria，就共享一个 $s(\mathcal C)$；若不同 prompts 生成了不同 criteria，则各自得到不同的 $s(\mathcal C_i)$。

Bare labels 的 embedding 信息有限，但不妨碍用冻结 judge 的评分行为估计 rubric-level ROIV。为验证 $s(\mathcal C)$ 确实有作用，实验使用自然完整 rubrics，不构造 1D/2D/3D 子集；固定 rubric 产生常数，prompt-specific generator 则产生 rubric-level variation。

## 5. 是否还使用 text embedding

Text embedding 保留为低成本 baseline / fallback，而不是主 $s$。如果资源限制导致无法生成 calibration probes，则定义

$$
\mathcal I_{\rm sem}(\mathcal C_g)
=\frac12\log\det\!\left(
I_{K_g}+\lambda W_g^{1/2}E_gE_g^\top W_g^{1/2}
\right),
$$

其中 $E_g$ 必须编码完整的 “criterion + reference/guidance + score anchors”，不能只编码 “helpfulness” 这样的 bare label。Encoder 冻结且全实验固定；可用开源的 Qwen3-Embedding-0.6B，并用另一 encoder 做排序稳定性检验。该版本只能称为 **semantic information / non-redundancy proxy**，不能声称识别了 operational completeness 或 unmeasured-confounding amount。

因为 covariance 是在每个 prompt 内用 $M$ 个 responses 单独估计，其秩最多为 $M-1$。若希望在单个 prompt 内有机会覆盖全部 $K_{g\ell}$ 个方向，需要 $M\ge K_{g\ell}+1$；$L$ 只能减小跨 prompt 平均的估计方差，不能提高某一 prompt 内 covariance 的秩。主实验固定 $M=16$，报告 $K_{g\ell}>M-1$ 的比例，并做 $M=8/32$ 敏感性；对满足 $K_{g\ell}\le M-1$ 的 prompts，建议检查

$$
M\ge \max(8,K_{g\ell}+1).
$$

增大 $L$ 可改善系统级均值/CI，增大 $M$ 才能改善单 prompt 内的秩与 covariance 估计。这些 probes 是一次性的自动 calibration sidecar：不改变 rubric、不改变 preference pair，也不训练新的网络。



# 距离表示——rubric-conditioned KL ambiguity

对 rubric 系统 $g$ 下的每个 preference pair，引入

$$
P^0_{gi}=\operatorname{Bernoulli}(\bar p_{gi}),
\qquad
Q_{gi}=\operatorname{Bernoulli}(q_{gi}).
$$

其中 $q_{gi}$ 是 adversary 认为 $y_{gi}^w\succ y_{gi}^l$ 的最坏情况概率。令 $\pi_g\ge0$、$\sum_g\pi_g=1$ 为各 producer/data track 的 mixture weight。本文对每个 pair 使用

$$
s_{gi}=s(\mathcal C_{gi})
$$

作为该 pair 所用 rubric 的距离权重；不在数据集内部归一化 $s$，也不根据 $s$ 改写 $\rho$。相同 `rubric_id` 必须读取同一个缓存值。

因为 $\bar p_{gi}$ 已 clip 到 $[\varepsilon,1-\varepsilon]$，Bernoulli KL 的最大值不超过 $-\log\varepsilon$。先定义无量纲 label distance

$$
d_{g,s}^{\rm label}(\mathbf Q,\mathbf P^0)
=\frac{1}{-\log\varepsilon}\frac1{n_g}
\sum_{i=1}^{n_g}s(\mathcal C_{gi})D_{\rm KL}(Q_{gi}\|P_{gi}^0)\in[0,1],
$$

再定义 rubric-weighted ambiguity set

$$
\mathcal U_{\rm KL}
=\left\{\mathbf Q:
\sum_{g=1}^{G}\pi_g d_{g,s}^{\rm label}(\mathbf Q,\mathbf P^0)
\leq\rho
\right\}.
$$

其中 $\rho$ 从始至终是一个固定常数。只有当所有样本共享同一个 rubric $\mathcal C_g$ 时，约束才可写成

$$
s(\mathcal C_g)d_g^{\rm label}(\mathbf Q,\mathbf P^0)
\le \rho.
$$

因此当 $\rho$ 固定时，

$$
d_g^{\rm label}(\mathbf Q,\mathbf P^0)
\le \frac{\rho}{s(\mathcal C_g)}.
$$

后一式只适用于全局固定 rubric，用来解释负相关关系，$\rho$ 本身没有变化。prompt-specific 情形必须保留逐项的 $s(\mathcal C_{gi})D_{{\rm KL},gi}$，不能用平均 $s$ 或单个 effective radius 代替。同一 rubric 下所有样本仍共享同一个 $s(\mathcal C)$。

本文把 $\mathbf P^\star\in\mathcal U_{\rm KL}(\mathbf P^0,\rho,s)$ 明确写成 sensitivity assumption。$s(\mathcal C_{gi})$ 决定各 rubric 信息与对应距离贡献之间的负相关关系，$\rho$ 始终保持固定；ROIV 本身不证明真实偏好分布必然位于该集合中。

# Minimax Objective

令 $A_{gi}=1$ 表示 $y_{gi}^w$ 在潜在真实偏好下仍应获胜，$A_{gi}=0$ 表示方向反转，并记对应损失为 $\ell_{gi}^+(\theta)$ 与 $\ell_{gi}^-(\theta)$。目标为

$$
\min_\theta\sup_{\mathbf Q\in\mathcal U_{\rm KL}}
\sum_g\frac{\pi_g}{n_g}\sum_i
\left[q_{gi}\ell_{gi}^+(\theta)+(1-q_{gi})\ell_{gi}^-(\theta)\right].
$$

其中

$$
D_{\rm KL}(Q_{gi}\|P^0_{gi})
=q_{gi}\log\frac{q_{gi}}{\bar p_{gi}}
+(1-q_{gi})\log\frac{1-q_{gi}}{1-\bar p_{gi}}.
$$

令 $K_\varepsilon=-\log\varepsilon$。inner maximization 可由一个全局 dual variable $\eta>0$ 精确求解：

$$
\min_{\theta,\eta>0}
\eta\rho
+\eta\sum_g\frac{\pi_g}{K_\varepsilon n_g}\sum_i s(\mathcal C_{gi})
\log\!\left[
\bar p_{gi}e^{K_\varepsilon\ell_{gi}^+/(\eta s(\mathcal C_{gi}))}
+(1-\bar p_{gi})e^{K_\varepsilon\ell_{gi}^-/(\eta s(\mathcal C_{gi}))}
\right].
$$

对应的最坏 preference probability 为

$$
q_{gi}^\star
=\sigma\!\left(
\operatorname{logit}(\bar p_{gi})
+\frac{K_\varepsilon[\ell_{gi}^+(\theta)-\ell_{gi}^-(\theta)]}{\eta s(\mathcal C_{gi})}
\right).
$$

所以 $s(\mathcal C_{gi})$ 的作用非常单一：作为 rubric-level weight 与对应 pair 的 KL 距离贡献相乘，在固定 $\rho$ 下产生负相关约束。它可能随数据行变化，但唯一原因是该行 rubric 不同；它不是由样本 loss 学出的 weight，也不改变 $\rho$ 或 $\bar p_{gi}$。

上述 Bernoulli preference-label objective 在 $\rho=0$ 的 primal 中精确恢复共享同一 $P^0$ 的 nominal stabilized MMPO；只有 clipping count 为 0 时才恢复未裁剪原 MMPO。它不会精确恢复原始 ODPO 或本文的 Scaled-DPO normalized-gap transfer。后两者若要作为 `+ours` consumer-transfer 实验，使用保持对应 nominal loss 不变的 **fixed-inner-batch loss-DRO**。固定 $B_{\rm DRO}=8$，令 $p_{gbi}=1/B_{\rm DRO}$、$t_{gbi}=q_{gbi}/p_{gbi}$、$\phi(t)=t\log t-t+1$，定义逐 rubric 加权的 generalized-KL distance

$$
d_{gb,s}^{\rm loss}
=\frac1{\log B_{\rm DRO}}\sum_{i\in b}p_{gbi}s(\mathcal C_{gbi})\phi(t_{gbi})\in[0,1].
$$

当 batch 内 rubric 相同，它退化为 $s_gD_{\rm KL}(Q\Vert P)/\log B_{\rm DRO}$；rubric 不同时不能把平均 $s$ 提到 KL 外。目标为

$$
\min_\theta\sup_{Q:\,\mathbb E_b\sum_g\pi_gd_{gb,s}^{\rm loss}(Q_{gb},P_{gb}^{\rm emp})\le\rho}
\mathbb E_b\sum_g\pi_g\mathbb E_{i\sim Q_{gb}}[L^B_{gbi}(\theta;m_{gbi})].
$$

令 $C_B=\log B_{\rm DRO}$。其精确 convex dual 为

$$
\min_{\theta,\eta>0}\eta\rho
+\mathbb E_b\sum_g\pi_g\min_{\nu_{gb}}
\left\{\nu_{gb}+\sum_{i\in b}p_{gbi}\frac{\eta s_{gbi}}{C_B}
\left[\exp\!\left(\frac{C_B(L^B_{gbi}-\nu_{gb})}{\eta s_{gbi}}\right)-1\right]\right\},
$$

其中 $s_{gbi}=s(\mathcal C_{gbi})$，$\nu_{gb}$ 由稳定的一维 root solve 决定，使

$$
q^\star_{gbi}=p_{gbi}\exp\!\left(\frac{C_B(L^B_{gbi}-\nu^\star_{gb})}{\eta s_{gbi}}\right),
\qquad \sum_{i\in b}q^\star_{gbi}=1.
$$

此时 $\rho=0$ 的 primal 精确恢复原 ODPO/冻结的 Scaled-DPO normalized-gap transfer，$s=1$ 是 uniform minibatch loss-DRO。实现必须对 `rho==0` 显式返回 nominal batch mean；dual 的测试使用 `rho->0+` 极限与容差。两类距离都先用冻结上界归一化到 $[0,1]$，所以不新增第二个 $\rho$；但 label shift 与 minibatch sample-distribution shift 仍是不同对象。论文必须把后两者称为 baseline-preserving **minibatch** consumer extensions，不能冒充 full-dataset DRO；核心证据仍来自 MMPO，也不能把 adapter 称为 Scaled-DPO paper reproduction。





# Training Pipeline

1. 有公开产物时复用原 rubric-DPO 方法的 rubric generation/specification、score aggregation 和 pair mining；缺失代码、数据或闭源模型时，使用预注册的 local adapter 并在行名标注 provenance，不称原方法完整复现；

2. 对 rubric 系统 $g$ 中的每个 pair，从发布的 aggregate scores 或经验证的 criterion-score aggregation 先得到 $m^{\rm raw}_{gi}$；只在该 system 的 training positive raw gaps 上冻结 $q^{\rm train}_{0.95,g}$，再按前述公式得到 clipped normalized $m_{gi}$，validation/test 复用 train $q_{0.95,g}$。随后用
   $\bar p_{gi}=\operatorname{clip}(\sigma(\gamma m_{gi}),\varepsilon,1-\varepsilon)$
   构造 nominal preference distribution $P_{gi}^0$；

3. canonicalize 每个 criterion set 并生成 `rubric_id`。对全局 rubric，在共同 500-prompt calibration bank 上平均 prompt-level ROIV，得到一个共享 `s`；对 prompt-specific rubric，在每个实际 assignment prompt 的冻结 probe responses 上计算 `s(C_i)`，exact rubric 重复时合并 prompts并共享缓存。共同 500 prompts 仅作跨方法分布/parser/scorer diagnostic。所有 rubrics 使用同一 RubricARROW scoring/failure contract；Prometheus 只在固定 200 prompts 做 sensitivity。这一步不使用人工 preference labels，也不把最终 chosen/rejected 作为 `s` 输入；

4. MMPO 对每个 pair 使用 $s(\mathcal C_{gi})D_{{\rm KL},gi}$；ODPO / Scaled-DPO normalized-gap transfer extensions 使用逐 rubric 加权 generalized-KL。两类距离都除以冻结上界，$\rho$ 始终取同一个固定常数；不对 $s$ 做数据集内归一化，也不构造 rubric-specific budget；

5. MMPO 用上面的精确 dual objective 训练 policy；ODPO / Scaled-DPO normalized-gap transfer consumer extensions 使用各自的 outer loss-DRO dual，并分别通过 $\rho=0$ 退化测试。

因此实际数据表增加 `rubric_id` 与 `rubric_sufficiency=s(C)`。全局 rubric 的所有行引用同一缓存值；prompt-specific rubric 的不同行可以引用不同值。$m_{gi}$ 仍随 pair 变化，因为它负责“赢了多少”；$s$ 只随 criterion set 变化，因为它负责“这套量尺有多充分”。



# 增量式研究设计

## 1. 主验证：同一数据上的多个自然完整 rubric systems

本文不构造 1D/2D/3D，也不人工删除 rubric criteria。总 coverage matrix 仍按 **rubric source/producer** 分列：UltraFeedback、WildChecklists、HelpSteer2 structured-feedback adaptation、Auto-Rubric、OpenRubrics、Rubric-ARM、OnlineRubrics-local、RRD-local 与 frozen EvoLM。列是实验来源，不等于 `s` 的索引单位：UltraFeedback、HelpSteer2、Auto-Rubric-global 每列共享一个 rubric；WildChecklists 及其余 prompt-specific producers 每列包含多个 $\mathcal C_{gi}$ 与多个 $s(\mathcal C_{gi})$。

主矩阵每个 system 至少包含 producer-specific DPO、MMPO nominal、MMPO+uniform、MMPO+ours；ODPO / Scaled-DPO normalized-gap transfer 的 nominal 与 baseline-preserving loss-DRO extensions 作为 consumer transfer。不同 native data 或 method-native released-data 的绝对分数不能用于给 producer 排名，主比较始终是同一列内的 `ours - uniform`。截图的最终表必须把表头写成 `Rubric system / source`、补 HelpSteer2 列与 uniform 行，并将 `Scaled DPO` 改名为 `Scaled-DPO normalized-gap transfer`。

先分别训练每个 system 的 MMPO nominal、uniform-KL 和 ROIV-KL；再把六套 system 的等量 preference data 混合，并在一个全局 ambiguity constraint 中联合训练：

$$
\sum_g\frac{\pi_g}{n_g(-\log\varepsilon)}\sum_i
s(\mathcal C_{gi})D_{\rm KL}(Q_{gi}\Vert P^0_{gi})\le\rho.
$$

比较 nominal mixture、$s\equiv1$、正确 rubric-level ROIV、within-producer shuffled-`rubric_id -> s` 与 rank-reversed mapping。相同 rubric 的重复样本始终绑定同一个被分配值；global-rubric system 没有 within-system shuffle。这里所有方法共享同一个固定 $\rho$。

需要明确：只有当一次训练所有样本共享同一 rubric 时，ours 才与 uniform normalized-KL 使用 radius $\rho/s_g$ 数学等价。该诊断适用于 UltraFeedback、HelpSteer2、Auto-Rubric-global；WildChecklists requirements 与五个 prompt-specific automatic producers在同一数据内已有 rubric-level variation，不存在单个 scalar effective radius。

## 2. 增量接入真实 rubric-DPO：RLCF / WildChecklists

RLCF 仍是最自然的真实载体，因为论文/仓库提供 checklist generation、criterion judge、importance weights、pair mining、训练代码与评测。但官方 HF 的 51,071-row offline release 只有 `prompt/chosen/rejected/chosen_score/rejected_score/requirements`，没有独立 criterion-score columns；所以首轮是 **released-data reuse**，先用 `chosen_score-rejected_score` 构 $m^{\rm raw}$，再按统一 train-only $q_{0.95}$ 合同得到 $m$。完整 scoring-pipeline reproduction 只有在仓库另行链接的中间产物 revision/checksum 被验证后才成立。WildChecklists 的 `requirements` 是 prompt-specific rubric：对实际进入训练/验证的每个 requirements criterion set，在其来源 prompt 的冻结 probe responses 上计算 $s(\mathcal C_i)$；相同 exact requirements hash 共享缓存。另留出的 500 prompts 只作跨方法 calibration diagnostic，不生成一个错误的 $s_{\rm RLCF}$ 常数。

RLCF 的作用是验证方法可以作为 sidecar 增量接入。每条 pair 从其 `rubric_id` cache 读取 $s(\mathcal C_i)$ 并乘在对应 normalized-KL contribution 前；仍使用与其他实验相同的固定 $\rho$。

## 3. 必须比较的 baselines

| 组别 | 方法 | 要回答的问题 |
|---|---|---|
| 原方法 | RLCF + standard DPO | 新模块相对原 rubric-DPO 是否有效 |
| offset 强度 | RLCF + ODPO；其 `+ours` 使用 baseline-preserving loss-DRO | pair-specific score-gap offset是否已经足够，以及 $s$ sidecar 能否迁移 |
| sample-weight 强度 | RLCF + Scaled-DPO normalized-gap transfer；其 `+ours` 使用 baseline-preserving loss-DRO | 只让强偏好产生更大梯度是否已经足够，以及 $s$ sidecar 能否迁移 |
| soft-label 强度 | RLCF + MMPO | 收益是否仅来自把“赢多少”变成 nominal preference probability |
| rating-gap 强度 | RLCF + Rating-DPO | 直接匹配/平移到 rubric score gap 是否已经足够 |
| 统一鲁棒 | KL minimax，$s(\mathcal C)\equiv1$，$\rho$ 不变 | rubric-derived distance weight 是否必要 |
| 错配控制 | KL minimax，在 rubric IDs 之间 shuffle/rank-reverse `rubric_id -> s` mapping | 收益是否来自有意义的 rubric--s 对应关系 |
| 最近邻方法 | DPO-PRO | 相对固定半径的 preference-only DRO 是否新增价值 |
| 全分布鲁棒 | KLDPO / Dr. DPO | 与现有 robust-DPO 边界是否清楚 |
| 本文 | ROIV-conditioned KL minimax | rubric sufficiency 是否能校准 ambiguity geometry |

关键消融包括：criterion count、text-embedding log-det、ROIV；rubric-level shuffled/rank-reversed `s(C)`、$s\equiv1$；KL 与 $\chi^2$；固定常数 $\rho$ 的敏感性以及 $\lambda,\gamma,\epsilon_s,M$ 的敏感性；不同 judge/probe policy 下 rubric 排序的稳定性。shuffle 的单位是 `rubric_id`：若多个样本共享同一 rubric，它们必须一起移动；prompt-specific unique rubric 时，数据行看似被 shuffle，但语义仍是错配 rubric 与其 $s$。

需要明确区分两种量：$m_{gi}$ 是当前 rubric 对第 $i$ 个 pair 测得的 preference magnitude；$s_{gi}=s(\mathcal C_{gi})$ 是该 criterion set 在共享冻结 calibration instrument 与其 assignment population 下的 operational sufficiency proxy。ODPO、Scaled-DPO normalized-gap transfer、MMPO 与 Rating-DPO 消费前者；本文把后者乘在对应 robust interface 的分布距离贡献前，$\rho$ 不变。

## 4. 共同 benchmark 与具体配置

现有文献没有一个数据集能让所有 rubric/strength-aware 方法的已报告结果直接横向比较。最成熟的 synthetic score-gap 共同交集是 UltraFeedback；RLCF/WildChecklists 最适合展示对现有 rubric-DPO pipeline 的增量接入；HelpSteer2 同时提供五维人类 ratings 和独立 preference strength；同一 WildChat response bank 上的多个 automatic-rubric producers 最适合识别 rubric-conditioned weighting。因此采用互补的四轨设置。

### Track A：RLCF / WildChecklists（真实 pipeline transfer）

首轮使用 WildChecklists 官方 HF released pairs，不声称复现全部上游 generation/scoring pipeline。发布的 aggregate scores 已在 $[0,1]$，直接定义

$$m_i^{\rm raw}=\texttt{chosen\_score}_i-\texttt{rejected\_score}_i.$$

随后严格应用前述 train-only $q_{0.95}$ normalization 与 tie/near-tie 规则；发布分数已在 $[0,1]$ 并不意味着跳过跨 system 的统一 margin 尺度合同。

所有 baseline 使用相同的 `requirements`、responses、chosen/rejected pairs 和训练预算。先在统一 trainer 中做 released-data DPO baseline run；只有另行链接的 criterion-level/scoring intermediates 被下载并验证 revision/checksum 后，才增加 full-pipeline reproduction。GPU 型号、显存与卡数暂不写死；每次运行记录实际资源。论文 controlled main table 默认统一使用 BF16 full-parameter fine-tuning（FSDP full-shard/ZeRO-3）；LoRA 仅作 smoke 或单独的参数高效 ablation。所有比较行必须共享相同 model、参数更新方式、epoch、effective batch 和 max length，完整配置以根目录 `PLAN.md` 为准。

- ODPO 使用作者代码的 $\alpha\log(m_i)$ 作为 log-gap offset（位于 $\beta$ 外），并只在 validation 上搜索 $\alpha$；
- Scaled-DPO normalized-gap transfer 使用 $w_i=m_i/\mathbb E_{train}[m]$，保持训练集平均权重为 1；HelpSteer2-Preference 已定义 Scaled DPO，但这里用 UltraFeedback normalized gap 替代论文原生强度，因此是受控 transfer；
- MMPO 使用 $\bar p_i=\sigma(\gamma m_i)$；
- Rating-DPO 搜索 rating-trust 参数 $\beta_1$；
- 在 UltraFeedback development split 上用 uniform-KL 选择一次 $\rho$，随后 uniform-KL 与本文在所有轨道共享该固定常数。

最关键的 transfer 表顺序为：Base、RLCF+DPO、RLCF+ODPO、RLCF+Scaled-DPO normalized-gap transfer、RLCF+MMPO、RLCF+Rating-DPO、RLCF+MMPO+uniform-label-DRO、RLCF+MMPO+ROIV-label-DRO；ODPO/adapter 的 loss-DRO uniform/ours 另以 consumer-extension 行报告。RLCF 每个 exact requirements rubric 有自己的 $s(\mathcal C_i)$；做 shuffled/rank-reversed control 时打乱 `rubric_id -> s` mapping，而不是任意打乱共享同一 rubric 的数据行。每个 consumer 内的 uniform 与 ours 使用同一个固定 $\rho$。

### Track B：UltraFeedback（成熟 strength-aware 算法共同实现 / 受控对比）

对齐 raw UltraFeedback 的四个 aspect annotations 与 `HuggingFaceH4/ultrafeedback_binarized` `train_prefs` 的同一批 pairs。冻结的 raw revision 必须包含官方 2023-12-29 `overall_score` 修复；本文仍由 instruction-following、truthfulness、honesty、helpfulness 四项重新聚合 response score，而把 `overall_score` 仅用于 source-alignment audit，并定义

$$m_i^{\rm raw}=\bar r_i^w-\bar r_i^l.$$

主表对 raw gap 应用前述 train-only $q_{0.95}$ normalization，并移除 raw ties 与 normalized near-ties；raw ties 只作 diagnostic appendix。为与 RLCF 和 automatic-rubric 轨道统一，policy/reference 固定为同一 revision 的 `Qwen/Qwen3-8B`，主轨 chat template 固定 `enable_thinking=false`。比较 DPO、ODPO、MMPO、Scaled-DPO normalized-gap transfer、Rating-DPO、ML-RDPO 和统一 robust controls；评测 AlpacaEval 2 LC、Arena-Hard、WildBench，RewardBench 或 held-out implicit-reward ranking accuracy 仅作为 calibration diagnostic。

UltraFeedback 的 instruction-following、truthfulness、honesty、helpfulness 四个方面对所有样本固定，因此完整 rubric 对整套数据只产生一个 $s_{\rm UF}$。不再构造低维 rubric 子集。该轨道比较 DPO、MMPO 的 nominal/uniform-label-DRO/ROIV-label-DRO，以及 ODPO/Scaled-DPO normalized-gap transfer 的 nominal/uniform-loss-DRO/ROIV-loss-DRO。核心 label-ambiguity 结论只由 MMPO 行承担。

### Track C：HelpSteer2（human rating / preference 双信号）

使用论文完整标注指南中的 helpfulness、correctness、coherence、complexity、verbosity 五维 ratings，并冻结 NVIDIA model card 推荐的有向聚合

$$r_{\rm HS2}=0.65h+0.80c+0.45\,coh+0.55\,comp+0.40(4-v).$$

该写法与 NVIDIA model card 的 $-0.40v$ 只相差常数，pair gap 完全相同；同时可在 ROIV 中使用反向归一化分数 $v^-=1-v/4$ 和正 importance $0.40$。这组 model-card weights 原用于 predicted attributes；本文把它迁移到 human ratings，是 structured-feedback scalarization adaptation，不是数据集发布的原生 margin。主 pair 明确设为 $y^w=\arg\max_y r_{\rm HS2}(x,y)$、$y^l=\arg\min_y r_{\rm HS2}(x,y)$，再令 $m_i^{\rm raw}=r(y^w)-r(y^l)$ 并应用同一 train-only normalization。不能把五个分数同向求和；另做仅含前三项的 goodness sensitivity。最终 HelpSteer2-Preference auxiliary release 含 7,118 pairs（6,766/352），有符号 strength 仅为 $\{-3,-2,-1,1,2,3\}$；原始 $-100$ 和聚合 0 均已排除。它只作为 held-out validity diagnostic，不参与 $s$、pair 定向或主 rubric margin 的构造。该轨道运行 DPO、MMPO 核心 objective，以及 ODPO/Scaled-DPO normalized-gap transfer 的 baseline-preserving loss-DRO transfer。

### Track D：Automatic rubrics（六个自然多-system 主识别）

固定同一批 English/non-toxic/deduplicated WildChat prompts、同一 policy 的 candidate responses和同一个 reference-free RubricARROW common grader，分别运行 Auto-Rubric paper global Theme--Tips rubric、OpenRubrics official generator、Rubric-ARM official generator、OnlineRubrics-local frozen-policy elicitation pipeline、RRD-local（EvoLM authors' reimplementation）与 frozen EvoLM official checkpoint。Auto-Rubric 主轨全数据共享一个 global rubric 和一个 $s_{\rm AR}$；其余五者通常逐 prompt 生成不同 criteria，因而逐 exact rubric 计算 $s(\mathcal C_{gi})$。WildChecklists 500-prompt bank只报告六个 producers 的 `s` 分布/parser/scorer diagnostic；正式训练权重在各自 WildChat train/validation assignment prompts 上计算。先固定 MMPO 比较 nominal、uniform 和 ours，再做 rubric-level shuffled/rank-reversed mapping。

主 controlled track 不调用商业 API。OpenRubrics、Rubric-ARM、EvoLM 使用发布 checkpoint；Auto-Rubric 使用论文 Appendix K 的 HelpSteer3-source global rubric，官方 HF query-specific annotations 仅在原 prompts 上做 artifact check；RRD 使用 EvoLM 作者的后续复现。

OnlineRubrics-local 先在独立 500-prompt induction split 上用冻结 `Qwen/Qwen3-8B`（`enable_thinking=false`）另行采样 `M_induction=8`（temperature 0.7, top-p 0.9, 独立 seed/hash），再用 synthetic seed rubrics 构建 induction-only pairs 做一轮 nominal-MMPO warm-up，以冻结不同的 current/control policy snapshots与 Qwen2.5-32B extractor。随后在**每个 common/calibration prompt**上独立生成 $C_0(x)$，current/control 各采 8 个 rollouts并按 sample index 构 8 对 contrasts，提取 $C_e(x)$，最终使用 $C(x)=\operatorname{dedup}(C_0(x)\cup C_e(x))$。induction prompt 的 rubric 文本不能跨 prompt 套用，辅助 rollouts也不能混入 candidate comparison。

RRD-local 使用冻结且彼此独立的 Qwen2.5-32B proposer（temperature 0/top-p 1/max-new-tokens 4096）和 Qwen3-1.7B filter/item judge（关闭 thinking，temperature 0/top-p 1/max-new-tokens 1024），并冻结 EvoLM 仓 `rrd_wu` overlay/revision。shared Qwen2.5-32B reference 作为 strong reference，冻结 Qwen2.5-1.5B-Instruct 生成独立 weak reference，两者不进入 candidate pool。controlled 主表将 RubricARROW 概率 argmax $1[p_{true}\ge p_{false}]$ 定义为 binary satisfaction，在每个 prompt-specific recursive rubric 的二值 matrix 上重算 WU，再得到 $m_i$ 和 $s_{\rm RRD}^{common}(\mathcal C_i)$；该行名为 `RRD-local/common-binarized-WU adapter`。Qwen3-1.7B native judge只作敏感性并得到单独的 rubric-level weights，禁止混用。

原生性边界必须写入表脚：OnlineRubrics 原 Generalist/Expert 数据与代码未公开，因此这里只能叫 local DPO adaptation；RRD 原文是 WildChat-4K + Dr.GRPO，本文是 approximate DPO adaptation；EvoLM frozen checkpoint 只叫 official-artifact transfer，不声称完整 co-evolution。原方法 native data/judge 结果另表报告，不能与 common-WildChat table 混为同一公平比较。

JudgmentBench、HelpSteer2-Preference、RewardBench2 或 JudgeBench 作为已有 human-preference labels 的独立 measurement diagnostic：评估每套完整 rubric system 的 pairwise agreement/AUC/NLL，不用于构造或补充 rubric，也不新增人工标注。JudgmentBench 的价值在于同一输出池同时有 expert rubric scores 和 expert comparative judgments，可直接验证本文的问题前提。

### 统计报告

全部方法先单 seed 做 coverage。九-source 核心主表对 producer-specific DPO、MMPO nominal、MMPO+uniform-label-DRO 与 MMPO+ours 运行 3 seeds；ODPO / Scaled-DPO normalized-gap transfer 只在预注册代表 sources 的 consumer-transfer 表运行 3 seeds。相同实验固定初始化、pair order、reference、optimizer、batch、长度和调参预算。全局 rubric 报单个 $s$/CI；prompt-specific producer 报 rubric-level `s(C_i)` 分布、response-seed stability、与独立 preference error/adversarial shift 的关系，以及 correct/shuffled/rank-reversed mapping 的差异。

## 5. 创新边界

Robust DPO 和 minimax 本身不是创新点。DPO-PRO 已经只对 preference distribution 做局部 DRO；KLDPO、WDPO、Dr. DPO 也已经覆盖不同形式的 distributional robustness。因此本文应把贡献写成：

> We introduce a rubric-level operational sufficiency statistic that weights the preference-distribution distance under a fixed ambiguity budget in existing rubric-based DPO pipelines.

更具体地说，新增点是 **rubric criterion set $\mathcal C\rightarrow s(\mathcal C)\times$ its distance contribution $\rightarrow$ fixed-$\rho$ ambiguity set**，而不是改变 $\rho$、DRO 或 rubric generation 本身。核心 MMPO 接口改变的是 preference-label ambiguity geometry；$s$ 可以因 rubric 不同而在数据行间不同，但它不是由样本 loss、margin 或 chosen/rejected 学出的 sample weight。



# 关键相关工作与方法选择

- RLCF: https://arxiv.org/abs/2507.18624 ，代码与预计算数据：https://github.com/viswavi/RLCF
- DPO: https://arxiv.org/abs/2305.18290 ，作者代码：https://github.com/eric-mitchell/direct-preference-optimization
- ODPO: https://arxiv.org/abs/2402.10571 ，作者代码：https://github.com/rycolab/odpo
- MMPO: https://arxiv.org/abs/2410.03145 ，作者代码：https://github.com/kykim0/margin-matching-pref-opt
- HelpSteer2-Preference（第 5.2 节与 Table 4 含 Scaled DPO；无独立官方 trainer）: https://arxiv.org/abs/2410.01257
- Rating-DPO / ML-RDPO: https://arxiv.org/abs/2602.00603
- 2D-DPO: https://aclanthology.org/2025.findings-naacl.455/
- SimPO: https://arxiv.org/abs/2405.14734
- DPO-PRO: https://arxiv.org/abs/2510.23590
- KLDPO / WDPO: https://arxiv.org/abs/2502.01930
- RRD: https://arxiv.org/abs/2602.05125
- RRD 可复用后续实现（非 RRD 原作者代码）: https://github.com/stellalisy/EvoLM
- PReMISE: https://arxiv.org/abs/2605.30803
- Auto-Rubric: https://arxiv.org/abs/2510.17314
- Auto-Rubric official data: https://huggingface.co/datasets/agentscope-ai/Auto-Rubric
- OpenRubrics: https://arxiv.org/abs/2510.07743 ，代码：https://github.com/wanghaoyu0408/OpenRubrics
- Rubric-ARM: https://arxiv.org/abs/2602.01511 ，代码：https://github.com/wanghaoyu0408/OpenRubrics/tree/main/rubric-arm
- OnlineRubrics: https://arxiv.org/abs/2510.07284
- EvoLM: https://arxiv.org/abs/2605.03871 ，代码：https://github.com/stellalisy/EvoLM
- HelpSteer2 data: https://huggingface.co/datasets/nvidia/HelpSteer2
- JudgmentBench data: https://huggingface.co/datasets/judgmentbench/JudgmentBench
- Prometheus local grader: https://github.com/prometheus-eval/prometheus-eval
- Prometheus-7B-v2 official prompt/model card: https://huggingface.co/prometheus-eval/prometheus-7b-v2.0
- Reference-free main scorer: https://huggingface.co/OpenRubrics/RubricARROW-8B-Judge
- RUBRIC-ARROW official probability-scoring code: https://github.com/Haoxiang03/RUBRIC-ARROW
- Qwen3 Embedding: https://arxiv.org/abs/2506.05176
- Visual rDPO: https://arxiv.org/abs/2604.13029

文献支持的判断是：

1. Auto-Rubric 的 embedding coding rate 用于压缩和去冗余已经诱导出的 rubrics，不等同于证明 rubric 完整；
2. PReMISE 直接从 per-criterion score matrix 的特征谱估计 effective dimensionality，更接近本文要测的 operational information；
3. RRD 发现 generic prompt-only rubrics 甚至可能降低 judge accuracy，并通过 rubric-score covariance 处理冗余；
4. 因此主指标采用 score-covariance log-det，text embedding 只作为 semantic baseline，是当前最稳妥的选择。
