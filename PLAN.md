# ICLR27 Unmeasured Rubrics：完整实验执行计划

> 本文件是当前实验的唯一主计划。旧的调研与计划文件只作为来源记录；若与本文件冲突，以本文件为准。

## 1. 研究问题与不可变合同

### 1.1 核心问题

现有 rubric-based preference optimization 用 rubric score gap

$$
m_{gi}=r_{\mathcal C_{gi}}(x_i,y_i^w)-r_{\mathcal C_{gi}}(x_i,y_i^l)
$$

表示 winner 相对 loser “赢了多少”。但当前 rubric 可能遗漏影响潜在偏好的因素，因此由 `m_i` 诱导的 nominal preference distribution 可能发生系统性 misspecification。

本文不估计遗漏维度，也不直接修改 `m_i`。令 $\mathcal C_{gi}$ 表示样本 $i$ 实际使用的 **criterion set**（包含 criterion wording 以及与该 criterion 绑定的 guidance/anchor/importance，但不包含 prompt、response、producer 名或 grader）。本文对每个不同的 rubric 计算

$$
s_{gi}=s(\mathcal C_{gi}),
$$

并将它乘在 preference-distribution distance 前，在固定 `rho` 下校准允许偏离 nominal preference distribution 的范围。若 $\mathcal C_{gi}=\mathcal C_{gj}$，则必须有 $s_{gi}=s_{gj}$；若整个数据集共享同一 criterion set，则退化为一个共享常数 $s_g$。prompt-specific generator 产生不同 criterion sets 时，则逐 rubric 得到不同的 $s_{gi}$。

### 1.2 用户约束与方法合同

- 不构造 1D/2D/3D 或人工删维度的 rubric。
- `s` 的索引单位是 rubric criterion set，而不是 producer/system，也不是 prompt 本身。相同 rubric 必须共享同一个 `s(C)`；不同 rubric 分别计算。于是 `s` 只会因为 rubric 改变而在样本间改变，不能直接使用 prompt、chosen/rejected、margin 或样本难度作为额外输入。
- 不新增人工标注。
- `m_i` 仍然随 pair 变化，负责表示“赢了多少”。
- `C_{gi}` 只指实际用于评分的 criteria 及其绑定的 guidance/anchor/importance；prompt、rubric generator、aggregation code 和全 systems 共享的 calibration grader/probe population都不属于 rubric 本体。producer/system 仅用于 provenance 与分组。
- `s(C_{gi})` 不乘到 `m_i`、不改写 soft label，也不是由样本损失学习出来的 weight；在 ODPO / Scaled-DPO normalized-gap transfer 的 outer-DRO extension 中，它只通过距离约束影响 adversarial distribution `Q`。
- `rho` 选择一次后，对所有 rubric systems、datasets、baselines、seeds 和 online rounds 保持同一个常数。
- robust learning 本身不是创新；贡献是 `rubric criterion set -> s(C) × distance -> fixed-rho ambiguity set`。
- 主对比统一从同一个 instruct/SFT checkpoint 开始，不为 ours 单独增加 SFT。

### 1.3 可支持与不可支持的结论

可支持的结论：

1. 在冻结 probe policy、rubric-assignment population 与 grader 后，ROIV 越高，该 criterion set 实际产生的稳定、非冗余评分信息越多。
2. 用该 rubric-level statistic 校准 ambiguity set，优于不鲁棒的 margin baseline 和 `s=1` 的 uniform robust baseline。
3. 核心 preference-label DRO 可以直接接入 MMPO；同一个 `s(C) × distance` sidecar 还可以通过一个保持原 loss 不变的 outer loss-DRO wrapper 接入 ODPO 与本文预注册的 **Scaled-DPO normalized-gap transfer**。两种接口必须分开命名，不能把后者写成与 MMPO 完全相同的 objective。

识别边界：只有当一次训练中的所有样本确实共享同一 rubric、因而 $s_{gi}\equiv s_g$ 时，weighted constraint 才与 uniform normalized-KL 使用 effective radius `rho/s_g` 数学等价。该边界适用于 UltraFeedback、HelpSteer2 和主轨的 Auto-Rubric global rubric；不适用于 prompt-specific 的 WildChecklists requirements、OpenRubrics、Rubric-ARM、OnlineRubrics、RRD 与 EvoLM。后者在同一数据内已经包含多个 rubric-specific weights，因此可以直接检验 rubric-conditioned geometry。

不能写成：

- `s` 恢复了真实遗漏维度；
- `s` 得到了每个 pair 的真实 margin；
- 单凭 rubric 文本或 ROIV 无假设地证明了全部人类偏好都已覆盖。

论文中的推荐表述是 **operational rubric sufficiency / measurement-information proxy**，不是 ground-truth completeness ratio。

## 2. 方法与实现接口

### 2.1 Nominal preference distribution

先对每个 rubric system 的 raw positive gaps 冻结同一套精确 normalization：在该 system 的 training split 上，对所有 `m_raw>0` 用 NumPy `quantile(method="linear")` 计算 `q95_g`，并定义

$$
m_{gi}=\operatorname{clip}\!\left(\frac{m^{\rm raw}_{gi}}{\max(q^{\rm train}_{0.95,g},10^{-8})},0,1\right).
$$

raw exact ties (`m_raw<=0`) 直接丢弃；归一化后 `m<0.05` 作为预注册 near-tie 丢弃。`q95_g`、裁剪数、tie/near-tie 数全部写入 manifest；validation/test 只应用 train 的 `q95_g`。这是 p95 scaling，不是待实现时再决定的 ECDF/quantile transform。然后对归一化 margin 使用

$$
\bar p_{gi}=\operatorname{clip}\bigl(\sigma(\gamma m_{gi}),\varepsilon,1-\varepsilon\bigr),
\qquad
P^0_{gi}=\operatorname{Bernoulli}(\bar p_{gi}).
$$

所有 systems 共享上述公式、`0.05` near-tie 阈值和 clipping 规则；只有各自 train split 的 `q95_g` 不同。

主配置冻结 `epsilon=1e-6`，并在每个数据 manifest 中记录触发 clipping 的样本数。nominal、uniform 与 ours 全部使用同一 `P^0`。若 clipping count 为 0，`rho=0` 恢复原 MMPO；若大于 0，论文行名必须写成 **stabilized/clipped MMPO**，不能声称逐点等于未裁剪的原公式。

### 2.2 Rubric Operational Information Volume

先把每个 rubric 写成 criterion-only 的 canonical JSON，并对 criterion text、绑定 guidance/anchors、方向和 importance 做 exact hash，得到 `rubric_id=r(C)`。prompt、response、producer 名和 grader 配置不进入这个 hash；不使用 embedding 把文字相近但不完全相同的 rubrics 强行合并。对每个不同 rubric `C_r`：

1. 令 `A_r` 为冻结 corpus 中被分配到该 exact rubric 的 prompt 集合。全局 rubric 的 `A_r` 是共享 calibration bank；prompt-specific rubric 通常只有其来源 prompt。若同一个 exact rubric 在多个 prompts 上重复出现，则合并这些 prompts，并最终共享一个 `s_r`。
2. 使用冻结 probe policy `Pi_0` 对每个 `x in A_r` 采样相同数量的 responses。prompt 只是 rubric 的应用/测量上下文，不属于 `C_r` 本身。
3. 在每个 prompt 内，用冻结 criterion grader 将 `M` 个 response 的 criterion scores 归一化到 `[0,1]`，得到 `Z_{r\ell}\in R^{M\times K_r}`。所有 criteria 先统一成“越高越好”的方向；有负聚合系数的维度先反向分数，再使用其正 importance。
4. **只使用 prompt 内 response covariance**，不把 prompt 难度的 between-prompt variance 当作 rubric 信息。对 `ell in A_r` 计算

$$
\Sigma_{r\ell}=\operatorname{Cov}_{j=1}^{M}(Z_{r\ell}[j,:]),
\qquad
G_{r\ell}=W_r^{1/2}\Sigma_{r\ell}W_r^{1/2}.
$$

5. 先得到每个应用 prompt 的 information volume，再只在 exact-rubric assignment set `A_r` 内求均值：

$$
\mathcal I_{\rm op}(\mathcal C_r)
=\frac1{|A_r|}\sum_{\ell\in A_r}\frac12\log\det(I+\lambda G_{r\ell}),
\qquad
s_r=s(\mathcal C_r)=\operatorname{clip}\left(1-e^{-\mathcal I_{\rm op}(\mathcal C_r)},\epsilon_s,1\right).
$$

主配置：scores 归一化到 `[0,1]`，`lambda=1`，`epsilon_s=0.05`。`lambda` 只做全局敏感性分析，不为某个 rubric 单独选择。

固定 rubric 在多个 prompts 上共享一个 `s_r`；prompt-specific rubric 则通常逐 prompt 得到一个 `s_{gi}=s(C_{gi})`。训练表同时保存 `rubric_id` 与查表得到的 `rubric_sufficiency`。同一 rubric 的重复实例必须复用缓存值；不同 rubric 不能先按 producer 求平均再用一个 system-level scalar 替代。这里的 `s(C)` 是 rubric 在其冻结 assignment population 下的 operational statistic，并非不依赖环境的 text-only 属性；环境在记号中被条件化掉，但 prompt 仍不属于 rubric 定义。

### 2.3 Calibration bank 的固定配置

- 从 WildChecklists 官方 51,071-row release 按固定 hash 留出 500 prompts，作为跨方法 `s` 分布、parser 和 scorer 的共同 calibration diagnostic，并从政策训练中移除。它不再被误写成“每个 producer 只有一个 `s`”的来源。
- 对实际进入训练/验证的每个 rubric assignment 生成冻结 probe response bank；全局 rubric 在共享 500 prompts 上估计一个 `s`，prompt-specific rubric 在其来源 prompt 的 probe responses 上估计自己的 `s`。同一 prompt 上不同 producers 复用相同 responses，以隔离 rubric 差异。
- 主配置固定 `M=16`；`K_r>M-1` 不会使 log-det 无定义，但 covariance rank 最多为 `M-1`，因此报告该比例并做 `M=8/32` sensitivity。若资源 pilot 要把正式主配置改为 `M=8`，必须在任何正式 `s` 或 preference corpus 生成前全局冻结，不能按 producer 改变。
- probe policy：与主 policy 起点相同的 `Qwen/Qwen3-8B` 冻结 revision；所有主轨 probe/candidate generation 固定 `enable_thinking=false`。
- generation：temperature `0.7`、top-p `0.9`、固定 max-new-tokens；保存 seed 与完整 generation config。
- 主 criterion grader 预先固定为 reference-free `OpenRubrics/RubricARROW-8B-Judge` 的冻结 revision，本地 vLLM、temperature `0`、`enable_thinking=false`，严格使用官方 JSON prompt与官方仓的 completion-logprob extractor，冻结 `top_logprobs=10`。在每个 `criteria_met_i` 位置读取 literal ` true`/` false` 的 log-prob，令 `p_i^T=exp(l_true)`、`p_i^F=exp(l_false)`，并把官方 probability score `p_i^T-p_i^F` 仿射到 `Z_i=clip((1+p_i^T-p_i^F)/2,0,1)`。这一模型对非 RubricARROW-native rubric text 的使用是所有 systems 共享的 common-scorer adapter，不冒充 producer-native judge。
- parser 合同固定：按 criterion index 读取官方 `explanation_i/criteria_met_i`，并用官方 extractor 对齐 boolean token 的 top-logprobs；JSON/索引或任一 `p_i^T/p_i^F` 缺失都把该 criterion 预注册记 `Z_i=0` 并计入 parser-failure rate，不重新调用模型。不得因为一个 producer 失败而改变其他 producers 的 prompt/response bank。producer-level criterion parse success `<99%` 时该 producer 不进主表；单个 rubric 的失败及其样本过滤必须进入 manifest。
- `prometheus-eval/prometheus-7b-v2.0` 只作预先 hash 固定的 200-prompt scorer sensitivity。该子集对所有 systems 相同；严格使用官方 absolute prompt、shared Qwen2.5-32B reference 与文档冻结的通用 1--5 anchors，分数映射 `(score-1)/4`。reference 与 candidate/Online auxiliary/RRD weak response 隔离；Prometheus 结果不进主 `m/s`，因此也不按 reference 兼容性筛选 calibration population。
- 对全局 rubric 在 prompts 上 bootstrap；对只有一个来源 prompt 的 rubric 在 responses/独立 probe seeds 上 bootstrap。报告 producer 内 `s(C)` 的分布、均值、分位数和不确定性，而不是只报告一个 system-level number。
- 在额外 100 个 prompts 上重复 3 次评分，估计 judge noise；noise-subtracted ROIV 仅作消融。

### 2.4 核心 objective：MMPO 上的 preference-label DRO

先把 Bernoulli KL 冻结为 `[0,1]` 尺度：

$$
d^{\rm label}_{g,s}(Q,P^0)
=\frac{1}{-\log\varepsilon}\frac1{n_g}\sum_i
s(\mathcal C_{gi})D_{\rm KL}(Q_{gi}\Vert P^0_{gi}),
\qquad 0\le d^{\rm label}_{g,s}\le1.
$$

这里的上界来自 `P^0` 已 clip 到 `[epsilon,1-epsilon]`。核心 ambiguity set 为

$$
\mathcal U_{\rm KL}=
\left\{Q:
\sum_g\pi_g d^{\rm label}_{g,s}(Q,P^0)\leq\rho
\right\}.
$$

训练目标：

$$
\min_\theta\sup_{Q\in\mathcal U_{\rm KL}}
\sum_g\frac{\pi_g}{n_g}\sum_i
\left[q_{gi}\ell^+_{gi}(\theta)+(1-q_{gi})\ell^-_{gi}(\theta)\right].
$$

对 `rho>0` 使用精确 dual/log-sum-exp 实现，只需把原 dual 中的 system-level `s_g` 替换为每项自己的 `s(C_{gi})`；对 `rho==0` 显式绕过 dual optimizer并直接返回 nominal loss，因为 dual 的最优值通常只在 `eta->infinity` 的极限达到。训练数据保存 `rubric_system_id`、`rubric_id` 和由 rubric cache 查得的 `s(C_{gi})`。

该目标在 `rho=0` 的 primal 中令 `Q=P^0`，精确恢复共享同一 clipped target 的 nominal MMPO；只有 manifest 证明 clipping count 为 0 时才称为原 MMPO。它**不会**在 `rho=0` 时恢复原始 ODPO 或 Scaled-DPO normalized-gap transfer：ODPO 是 hard-label loss 中加入 margin offset，adapter 是用 margin 缩放 hard-label DPO loss。因而不能把同一个 Bernoulli objective 未经修改地命名为 `ODPO+ours` 或 `Scaled-DPO normalized-gap transfer+ours`。

### 2.5 ODPO / Scaled-DPO normalized-gap transfer 的 baseline-preserving extension

为检验 `s(C)` 是否可以增量接入其他 margin consumers，同时保持对应 nominal loss 的定义，对 `B in {ODPO, Scaled-DPO normalized-gap transfer}` 使用 outer loss-distribution DRO。ODPO 的单样本 loss 使用作者代码的 log-gap 形式 `softplus(-(beta*h-alpha*log(gap)))`，其中 offset 位于 `beta` 外。HelpSteer2-Preference 第 5.2 节确实定义并实验了 Scaled DPO；UltraFeedback 行以 four-aspect normalized gap 代替论文原生的人工强度 `{1,2,3}`，因此只能称为 normalized-gap transfer，而不是原设置或官方代码复现。

全数据集 inner maximization 每步不可执行，因此正式实现预注册为 **fixed-inner-batch loss-DRO**：每个 system 的 dataloader 使用固定 `B_DRO=8` 个样本组成 inner batch `b`，再用 gradient accumulation 保持 effective batch 64。令 $p_{gbi}=1/B_{\rm DRO}$、$t_{gbi}=q_{gbi}/p_{gbi}$，并使用逐项非负的 generalized-KL generator $\phi(t)=t\log t-t+1$。rubric-weighted minibatch distance 定义为

$$
d^{\rm loss}_{gb,s}(Q_{gb},P^{\rm emp}_{gb})
=\frac1{\log B_{\rm DRO}}\sum_{i\in b}p_{gbi}s(\mathcal C_{gbi})\phi(t_{gbi}).
$$

当 batch 内所有样本共享同一 rubric 时，它精确退化为 $s_gD_{\rm KL}(Q\Vert P)/\log B_{\rm DRO}$；当 rubrics 不同时，不能先把 $s$ 求平均再移到 KL 外面。由于 $0<s\le1$ 且 $\phi(t)\ge0$，该距离仍在 `[0,1]`。对随机 batches 的平均约束定义

$$
\mathcal U^B_{\rm loss}
=\left\{\{Q_{gb}\}:\mathbb E_b\sum_g\pi_gd^{\rm loss}_{gb,s}(Q_{gb},P^{\rm emp}_{gb})\leq\rho\right\},
\qquad
\min_\theta\sup_{Q\in\mathcal U^B_{\rm loss}}
\mathbb E_b\sum_g\pi_g\mathbb E_{i\sim Q_{gb}}[L^B_{gbi}(\theta;m_{gbi})].
$$

这一无量纲化与固定 batch contract 对所有 systems/seeds 相同，只定义距离尺度，不改变 `rho`，也不引入第二个 robustness budget。论文必须称它为 minibatch loss-DRO extension，不能写成精确的 full-dataset DRO。

令 $C_B=\log B_{\rm DRO}$。当 $s_i$ 不同时，原来的单个 log-sum-exp 公式不再成立；离散经验分布下使用带一个 normalization multiplier 的精确 convex dual：

$$
\min_{\theta,\eta>0}
\eta\rho+\mathbb E_b\sum_g\pi_g\min_{\nu_{gb}}
\left\{\nu_{gb}+\sum_{i\in b}p_{gbi}\frac{\eta s_{gbi}}{C_B}
\left[\exp\!\left(\frac{C_B(L^B_{gbi}-\nu_{gb})}{\eta s_{gbi}}\right)-1\right]\right\},
$$

其中 $s_{gbi}=s(\mathcal C_{gbi})$，$\nu_{gb}$ 用稳定的一维 root solve 使下式归一化：

$$
q^\star_{gbi}=p_{gbi}\exp\!\left(\frac{C_B(L^B_{gbi}-\nu^\star_{gb})}{\eta s_{gbi}}\right),
\qquad \sum_{i\in b}q^\star_{gbi}=1.
$$

这里每个 `s(C)` 仍只乘在对应 rubric 的分布距离贡献前，`rho` 仍是所有实验共享的同一个常数，`m_i` 与对应 nominal ODPO/adapter loss 完全不变。`rho=0` 的 primal 中每个 `Q_gb=P^emp_gb`，因此分别精确恢复原 ODPO 与本文冻结的 Scaled-DPO normalized-gap transfer；代码对 `rho==0` 显式返回 nominal batch mean。`s=1` 是对应的 uniform loss-DRO control。两个接口的距离都已无量纲化到 `[0,1]`；尽管如此，MMPO 的 label shift 与 ODPO/adapter 的 minibatch sample shift 仍是不同对象。

论文中必须把两类结果分开表述：MMPO 行检验 rubric-induced **label/margin uncertainty**；ODPO/adapter 行只检验 rubric-conditioned statistic 作为 **baseline-preserving minibatch loss-DRO sidecar** 的 consumer transfer，不用它们替代核心 MMPO 理论证据。

## 3. 模型、资源与公平性配置

### 3.1 主模型与训练方式

- 暂不冻结 GPU 型号、显存容量或卡数；每次运行在 manifest 中记录实际 accelerator、数量、显存、driver/CUDA 与 wall-clock。正式结果只要求同一比较组共享训练方式和总优化预算。
- policy/reference：固定同一 revision 的 `Qwen/Qwen3-8B`；这是主模型的唯一身份，不再使用 Qwen2.5-7B-Instruct 或 Qwen3-4B 作为论文主表 policy。
- Qwen3 主训练、probe/candidate generation 与 alignment benchmarks 统一固定官方 chat template 的 `enable_thinking=false`；thinking-mode 只能作为单独 sensitivity，不能与 non-thinking 主表混合。
- 主表默认统一使用 **BF16 full-parameter fine-tuning**；`full-parameter` 指全部 policy 参数更新，模型状态可由 FSDP `FULL_SHARD` 或 DeepSpeed ZeRO-3 分片，不要求每张卡保留完整副本。
- 开启 FlashAttention、gradient checkpointing 与 FSDP full-shard/ZeRO-3；预计算并缓存 frozen reference 的 chosen/rejected log-prob，使正式训练无需常驻第二个 Qwen3-8B reference model。
- max sequence length `2048`，相同 chat template、tokenizer、padding/truncation。
- 初始训练预算：1 epoch；effective batch size `64`。显存不足时只调整 micro-batch 与 gradient accumulation，不改变 effective batch。
- ODPO / Scaled-DPO normalized-gap transfer 的 loss-DRO extension 是例外中的固定实现合同：`B_DRO=8` 是构造 adversarial distribution 的 inner batch，不得因显存调整；再累积 8 个 inner batches 得 effective batch 64。
- full-parameter learning rate 先在 `{5e-7, 1e-6}` 中用统一 validation budget 选择一次；若 pilot 显示两端均不合适，只能在所有比较行共享的同等预算内扩展一次网格。
- 主 seeds：`13, 42, 100`。执行分两阶段：先用 seed `42` 跑完四个 baseline 并完成本地 test；资源允许后再补 `13/100`。第一阶段只是可比的单 seed 证据，不报种子方差，不冒充最终三 seed 主表。

LoRA 只用于代码 smoke、低预算开发或可选的参数高效 ablation，不作为默认主表。若实际可用资源在 `max_length=2048` 的 full-parameter smoke 中仍无法通过，才可把**整个比较组**统一降级为 LoRA，并将其单独成表；不得让 baseline 与 ours 使用不同的参数更新方式。

### 3.2 `rho` 与其他超参数

- 在 UltraFeedback development split 上，使用 `MMPO + uniform robust (s=1)` 从预注册网格选择一个 `rho`。
- `rho` 一旦选定，立即冻结；ours 不重新调 `rho`。
- 不设置 `rho_label/rho_loss` 两套预算。第 2.4/2.5 节先把两类 KL 分别除以其冻结上界，使同一个 `rho` 表示 `[0,1]` divergence budget；consumer 间仍只比较各自的 `ours - uniform`，不比较 adversarial shift 的绝对大小。
- UltraFeedback MMPO baseline 的 `gamma` 固定为作者8B recipe的 `2.2`，不再消耗本地 pilot 预算重选；automatic-rubric producer 之间必须共享同一个 `gamma` 和 margin normalization。UltraFeedback development pilot 统一改为64 optimizer steps：仅搜索 DPO LR `{5e-7,1e-6}` 与 ODPO alpha `{0.1,0.5,1.0}`。NUS 共享节点限制同时最多2 GPU。实测后 pilot 冻结为 FSDP no-offload、`per_device=1`、`gradient_accumulation=32`、effective batch `64`；8-step peak reserved `77.37GB/GPU`、吞吐 `13.7s/step`。CPU offload 配置作为 OOM fallback，不用于主 pilot。
- ODPO 的 offset scale、Scaled-DPO normalized-gap transfer 的 weight normalization、MMPO 的 target mapping各自允许同等数量的 validation trials，但 ours 与其对应 nominal/uniform 行共享这些值。
- 所有方法共享 optimizer、scheduler、epoch、batch、max length、seed 和 checkpoint-selection rule。

## 4. 数据与角色划分

| 数据 | 角色 | 是否用于训练 | 需要缓存的字段 |
|---|---|---:|---|
| raw `openbmb/UltraFeedback` | 63,967 prompts × 4 completions；native four-aspect annotations | 是 | source、instruction、models、completions；completion 内缓存 model、response、annotations、overall_score |
| `HuggingFaceH4/ultrafeedback_binarized` `train_prefs` | 约 61.1k 公共 chosen/rejected alignment pairs | 是 | prompt、prompt_id、chosen、rejected、messages、score_chosen、score_rejected |
| WildChecklists released offline pairs | 51,071-row native checklist preference data；`requirements` 含 prompt-specific criteria/importance，`chosen_score/rejected_score` 为发布的 aggregate scores；先留出 500 prompts 作九-system common calibration | 是，但 calibration 500 不进政策训练 | prompt、chosen、rejected、chosen_score、rejected_score、requirements、split/hash |
| HelpSteer2 ratings | 21,362 条 native five-attribute human ratings；本文使用 model-card-derived scalarization adapter 构 pair/m | 是 | prompt、response、helpfulness、correctness、coherence、complexity、verbosity、split |
| HelpSteer2-Preference auxiliary file | 7,118 对独立人工方向/强度标注（6,766 train / 352 validation）；最终 strength 为 `{-3,-2,-1,1,2,3}` | 否，仅做 held-out validity/magnitude diagnostic | split、prompt、response_1、response_2、preference_strength、preference_statement、preference_elaboration |
| Auto-Rubric official data | 38,459 条 HelpSteer3-derived query-specific rubrics/pairs；只用于 released-data artifact check，不等于完整方法复现 | 否，不直接跨 prompt 套用 | unique_id、source、rubrics、rubric_valid、rubric_epoch、input、output、preferred、domain、language |
| Auto-Rubric paper global rubric | 论文 Appendix K 发布的 HelpSteer3/UltraFeedback Theme--Tips 全局 rubric；论文原 DPO 使用 HelpSteer3-source rubric 在 WildChat 标 pair | 是 | source rubric、digitization/checksum、judge outputs |
| OpenRubric-v2 | 74,214-row 官方 rubric/pair/judge 数据；来源含 UltraFeedback、Magpie、Skywork-Preference、Synthetic-IF、MegaScience、Medical-o1 | official released-artifact reuse / method-native sanity；full API curation另行 gated | instruction、response_a、response_b、winner、rubric、judge、source |
| OnlineRubrics Generalist/Expert | 原文分别为 human-written Generalist 与 Physics/Chemistry/Biology/Math Expert rubrics | 否；截至 2026-08-10 未公开 | 只记录论文统计与 unavailable 状态 |
| Tulu 3 preference mixture | EvoLM 原论文约 271k prompts 的训练来源 | 只做 provenance/必要的 method-native smoke | source、prompt id、dedup hash |
| WildChat fixed split | automatic-rubric controlled DPO 的 `4k train / 500 validation` common prompt pool，加 500 个不重叠 Online induction prompts；RRD 原文也使用 4k WildChat | 是 | prompt id、source、split、dedup hash |
| Shared reference-answer bank | 闭合 Prometheus sensitivity 的官方 absolute-grading 输入，并作为 RRD strong reference；不是 preference label | 否，仅供 sensitivity / RRD filter | prompt id、Qwen2.5-32B revision、answer、generation config、hash |
| RRD strong/weak reference banks | 完整运行 RRD misalignment filter；不得进入 candidate pool | 否，仅供 RRD rubric filter | prompt id、strong/weak model revision、answers、generation config |
| JudgmentBench | 同一输出池上的 expert rubric scores 与 expert pairwise judgments | 否，仅做 premise/measurement diagnostic | task rubric、item weights、rubric score、pairwise label、output id |
| RewardBench2 / 其他公开 human-preference validation | 独立 measurement diagnostic | 否 | pair、human/preference label、subset |
| specialized benchmarks | alignment/instruction-following evaluation | 否 | evaluator revision、judge config |
| five EvalScope general benchmarks | 最终通用能力评测 | 否 | task config、few-shot setting、seed |

所有数据必须记录 dataset revision、下载日期、license、原始 split 与最终过滤数量。training、rubric-induction、s-calibration、validation 和最终 evaluation prompts 必须去重。UltraFeedback 的本文 `m_raw` 由四个 aspect annotations 重新聚合；`overall_score` 只用于 source-alignment audit。冻结的 raw revision 必须包含官方 2023-12-29 overall-score 修复，不能把当前 revision 直接描述为仍含未修复错误。

HelpSteer2 的五个属性不能同向求和。主结果冻结 NVIDIA `Llama3-70B-SteerLM-RM` model card 推荐的完整五维标量化：

$$
r_{\rm HS2}=0.65h+0.80c+0.45\,coh+0.55\,comp+0.40(4-v),
\qquad m_i^{\rm raw}=r_{\rm HS2}(y_i^w)-r_{\rm HS2}(y_i^l),
$$

其中 `h/c/coh/comp/v` 分别为 helpfulness/correctness/coherence/complexity/verbosity，原始范围都是 0--4。`0.40(4-v)` 与 NVIDIA model card 的 `-0.40v` 只相差常数，所以 pair gap 完全相同，但所有 ROIV importance 都可以保持非负；计算 `s_HS2` 时对应使用归一化反向分数 `v^-=1-v/4`。主 pair 明确取 `chosen=argmax_y r_HS2(x,y)`、`rejected=argmin_y r_HS2(x,y)`，再令 `m_raw=r(chosen)-r(rejected)` 并按第 2.1 节归一化。另报告只用 `0.65h+0.80c+0.45coh` 的三项 goodness sensitivity；独立的 human preference direction/strength 只做效度检查，不反向参与 `s`、pair 定向或选择聚合权重。

该五维标量化把 NVIDIA reward-model card 针对 **predicted attributes** 推荐的权重迁移到 human ratings，是本文的 structured-feedback adaptation；它不是 HelpSteer2 数据集发布的原生 rubric margin，也不称 paper reproduction。

## 5. Baselines 与必做对照

### 5.1 Margin consumers

一个核心 consumer 与两个预注册 transfer consumers：

1. **MMPO**：用 `m_i` 构造 soft preference target。
2. **ODPO**：使用作者实现的 pair-specific log-gap offset `alpha*log(gap_i)`；UltraFeedback 受控适配令 `gap_i=m_i>0`。
3. **Scaled-DPO normalized-gap transfer**：用 `m_i/E_train[m]` 缩放 DPO sample loss/gradient；论文原生 Scaled DPO 使用人工强度 `{1,2,3}`，UltraFeedback 行是受控 transfer。

附录：**ML-RDPO**，代表显式 rating-gap matching。

代码 provenance 必须如实记录：[DPO](https://github.com/eric-mitchell/direct-preference-optimization)、[MMPO](https://github.com/kykim0/margin-matching-pref-opt)、[ODPO](https://github.com/rycolab/odpo) 均有作者公开代码；其中 MMPO 最接近本文数据接口，ODPO 的原入口不是通用聊天 preference corpus 的开箱即用实现，需要聊天数据 adapter。HelpSteer2-Preference 第 5.2 节与 Table 4 发布了 **Scaled DPO 公式和实验结果**，但没有独立、可直接运行的官方 Scaled-DPO trainer；本文以论文公式做数值对齐，并把 UltraFeedback 版本命名为 `Scaled-DPO normalized-gap transfer`。主工程以当前 TRL 为共同基础设施，避免安装不兼容的旧环境。

当前实施顺序冻结为：`DPO/MMPO/ODPO/Scaled DPO baselines -> 本地验证与评测 -> s_UF -> robust methods`。robust 阶段只预留 `kl` 与 `wasserstein_w1` 两个 backend 接口；在 Wasserstein ground cost、radius normalization、solver 与选择规则写回方法文档前，不实现或运行 robust 训练。

### 5.2 每个 margin consumer 的证据与名称

| 行 | 设置 | 回答的问题 |
|---|---|---|
| 0 | Base checkpoint | 未做 preference optimization 的起点 |
| 1 | Vanilla DPO | 只知道谁赢谁输的下界 |
| 2 | `B + m` | 原始 margin consumer 是否有效 |
| 3 | `B + m + uniform robust (s=1)` | 一般 minimax robustness 是否已足够 |
| 4 | `B + m + ours (s=s(C))` | rubric-conditioned sufficiency 是否带来额外价值 |
| 5 | `B + m + uniform robust (validation-tuned radius)` | 单系统中，逐数据集调半径的强 oracle control |

第 3 与第 4 行必须使用完全相同的固定 `rho`。第 2--4 行必须使用完全相同的 pairs 和 `m_i`。

第 5 行只用于诊断，允许 baseline 在每个数据集上单独调 radius，因此明确不是本文的 fixed-`rho` 方法。只有全数据共享同一 rubric 的 UltraFeedback、HelpSteer2 和 Auto-Rubric-global 另做数值测试：uniform robust 使用 `rho/s_g` 必须与 ours 完全相等。prompt-specific rubric 数据不存在单个 `s_g`，因此不得声称这种标量等价。

表中 `robust` 的具体接口随 consumer 冻结：MMPO 使用第 2.4 节的 preference-label DRO；ODPO 与 Scaled-DPO normalized-gap transfer 使用第 2.5 节的 baseline-preserving outer loss-DRO。表格标题、run id 和论文脚注都必须写清楚这一点。若没有实现并通过 `rho=0` 的精确退化测试，ODPO/adapter 的 `+uniform/+ours` 单元格保持 `not defined`，不能先跑一个 hybrid loss 再沿用原方法名。

## 6. Phase 0：仓库、数据与 loss preflight

### 6.1 收集并冻结

- TRL DPOTrainer 与 exact commit；DPO 作者仓只用于参考对齐。
- [DPO 作者实现](https://github.com/eric-mitchell/direct-preference-optimization)、[MMPO 作者实现](https://github.com/kykim0/margin-matching-pref-opt)、[ODPO 作者实现](https://github.com/rycolab/odpo)；Scaled-DPO normalized-gap transfer 只按 HelpSteer2-Preference 的 Scaled BT reward-model公式构造 DPO adapter，明确不存在可声称复现的 paper-specific Scaled-DPO trainer；ML-RDPO 作者实现放附录。
- RLCF/WildChecklists 代码与预计算数据。
- Auto-Rubric 官方数据与论文完整 rubric/prompts；论文链接到 RM-Gallery/OpenJudge 通用框架，但尚未验证其中包含论文完整 Propose--Evaluate--Revise/Theme--Tips pipeline，且没有已确认的论文生成 checkpoint。
- OpenRubrics 官方统一仓、OpenRubric-v2、RubricRM-v2 generator/judge；Rubric-ARM 官方代码与 8B generator/judge。
- EvoLM 官方代码与 EvoLM-8B checkpoint；同时冻结其 RRD-WU baseline reimplementation 的 commit。
- OnlineRubrics 论文 prompts 和 Scale 项目页；当前没有官方代码/公开训练数据，本文实现必须标记为 `OnlineRubrics-local/offline-adapt`。
- RRD 原论文与 prompts；当前没有 RRD 原作者代码，使用 EvoLM 作者的后续复现时必须标明 provenance。本地主轨冻结为两个独立 actor：`Qwen/Qwen2.5-32B-Instruct` proposer（temperature 0, top-p 1, max-new-tokens 4096）与 `Qwen/Qwen3-1.7B` filter/item judge（关闭 thinking，temperature 0, top-p 1, max-new-tokens 1024）；两者 revision 和完整 `rrd_wu` overlay 写入 manifest。
- RubricARROW-8B-Judge 主 pointwise grader 与官方 RUBRIC-ARROW code 的 exact revisions、官方 JSON prompt、`top_logprobs=10` extractor、`p_true-p_false` 映射与 parser contract。
- Prometheus-7B-v2 sensitivity checkpoint/exact revision；Qwen2.5-32B-Instruct reference/extractor revision与 Qwen2.5-1.5B-Instruct weak-reference revision。

### 6.2 必做单元测试

1. ODPO 在 offset scale 为 0 时等于 DPO。
2. Scaled-DPO normalized-gap transfer 在 weight 全为 1 时等于 DPO。
3. `rho==0` 代码分支直接返回 nominal loss；MMPO 逐样本等于共享同一 `P^0` 的 stabilized MMPO，ODPO / Scaled-DPO normalized-gap transfer 分别等于对应冻结的 nominal loss。
4. 对 dual 另测 `rho->0+` 的极限在数值容差内收敛到 nominal；不要求有限 `eta` 在 `rho=0` 取得极限。
5. ours 在 `s=1` 时逐样本等于对应 consumer 的 uniform robust。
6. chosen/rejected 交换后正负 preference branch 对称。
7. 对 `rho>0`，两类 dual objective 分别与小规模显式 inner maximization 数值一致。
8. 每个数据 manifest 记录 MMPO target clipping count；只有 count 为 0 才使用 `original MMPO` 标签，否则统一使用 `stabilized/clipped MMPO`。
9. canonical rubric JSON 完全相同的样本具有相同 `rubric_id` 并读取同一个 `s(C)`；rubric 不同则不能因 producer 相同而强制共享 `s`。
10. `s(C)` 不进入 `m_i`、原 baseline 的 sample weight 或 `rho` 的计算；它只进入 ambiguity-set distance coefficient，并可由 inner maximization 间接改变 adversarial `Q`。
11. per-rubric weighted generalized-KL dual 与小规模显式 inner solver 对齐；当 batch 内 `s_i` 全相同时退化为原 scalar-`s` log-sum-exp。

### 6.3 Smoke

用 UltraFeedback 1k pairs、单 seed、短训练跑：DPO、MMPO、MMPO+uniform、MMPO+ours-with-`s=1`。这一步只验证 ours/uniform 不变性，不使用正式 `s_UF`。检查 NaN、loss、policy-reference KL、reward margin、response length、dual variable 与最坏情况 `q*`。

**Gate 0：** baseline loss 数值对齐、robust invariants 全部通过、smoke 可恢复和可复现后，才生成全量 automatic-rubric corpus；允许在此前使用上述 1k/32-prompt 最小 pilot。

## 7. Phase 1：完整 rubric 的 `s` calibration 与有效性检查

### 7.1 每种方法实际产生多少个 `s`

| 数据/producer | Rubric 粒度 | 正式 `s` 单位 |
|---|---|---|
| UltraFeedback | 全数据固定四个 aspects | 一个 `s_UF` |
| WildChecklists/RLCF | 每个 prompt 的 `requirements` 通常不同 | 每个 exact requirements checklist 一个 `s(C_i)`；重复 checklist 共享缓存 |
| HelpSteer2 | 全数据固定五个 attributes | 一个 `s_HS2` |
| Auto-Rubric 主轨 | Appendix-K global Theme--Tips | 一个 `s_AR`；官方 query-specific release 另按每个 rubric 计算 artifact diagnostic |
| OpenRubrics | prompt-specific generated rubric | 每个 generated criterion set 一个 `s(C_i)` |
| Rubric-ARM | prompt-specific generated rubric | 每个 generated criterion set 一个 `s(C_i)` |
| OnlineRubrics-local | 每个 prompt 的 `dedup(C0(x) union Ce(x))` | 每个 prompt-specific rubric 一个 `s_t(C_i)`；online 每轮重新计算 |
| RRD-local | 每个 prompt 的 recursive rubric | 每个 recursive criterion set 一个 `s(C_i)`；common/native judge 两轨不混用 |
| frozen EvoLM | 每个 prompt 生成 JSON rubric | 每个 generated criterion set 一个 `s(C_i)` |

WildChecklists 官方 HF 没有 criterion-score 列；对进入训练的 requirements 及其冻结 probe responses用 shared grader 重新产生 criterion scores。RRD 主 controlled row继续用 RubricARROW `1[p_true>=p_false]` binary matrix重算 WU，native Qwen3-1.7B judge/WU 只作 sensitivity。所有 prompt-specific rubrics 在训练前冻结并 canonicalize；prompt 不进入 rubric hash。

### 7.2 不使用删维度的四类验证

1. **理论性质：** 给出 Gaussian measurement model 下 ROIV 等于 mutual information、对新增 PSD information 单调、对高度相关评分方向增益有限的命题与证明。
2. **估计稳定性：** bootstrap CI、split-half stability、不同 response sampling seeds 的稳定性。
3. **judge/probe robustness：** 在小规模子集上更换一次 probe seed 或备用 grader，报告 `s` 的排序与绝对值敏感性。
4. **独立测量效度：** 在已有 human-preference labels 的 JudgmentBench、RewardBench2/JudgeBench 和 HelpSteer2-Preference evaluation split 上评估 rubric 的 pairwise agreement、AUC/NLL；对 prompt-specific rubrics 检查 rubric-level `s(C_i)` 与独立 agreement/error 的关系。JudgmentBench 尤其用于比较同一批输出上的 expert rubric score 与 expert comparative judgment。这里不训练模型、不删 rubric 维度、也不新增人工标注。

`s` 是连续权重，不人为设定“合格/不合格”的阈值。全局 rubric 报告单个 `s` 与 CI；prompt-specific producer 报告 `s(C_i)` 的分布、分位数、response-seed stability及其与独立 agreement 的关系。`rho/s_g` 只对全局固定 rubric 有单一 effective-radius 含义。

**Gate 1：** 所有进入训练的 `s(C)` 有限；producer 内分布非退化且 response-bootstrap/seed stability 达标。全局 rubric 的 relative SE 建议小于 10%；unique prompt-specific rubric 主要检查 response bootstrap 与重复 probe seed。若不稳定，先增加 `M` 或修正 grader parsing。若 `s` 与独立 measurement validity 完全无关，论文只能声称 operational information，不能声称 sufficiency。

## 8. Phase 2：native-rubric / structured-feedback 数据上的主实验

### 8.1 UltraFeedback

1. 对齐 raw UltraFeedback 4-aspect scores 与 binarized chosen/rejected。
2. 使用完整 rubric 的官方/冻结 aggregation 得到 `m_raw_i`，并按第 2.1 节只用 train split 冻结 `q95_UF`。
3. raw ties 与 normalized near-ties 不进入任何主训练；可在不影响主配置的 diagnostic appendix 单独报告 raw ties。
4. 跑以下矩阵：

| 方法 | nominal | uniform robust | ours |
|---|---:|---:|---:|
| Vanilla DPO | yes | n/a | n/a |
| MMPO | yes | yes | yes |
| ODPO | yes | yes | yes |
| Scaled-DPO normalized-gap transfer | yes | yes | yes |
| ML-RDPO | appendix | optional | appendix |

### 8.2 WildChecklists/RLCF

1. 第一轮直接使用 HF 发布的 51,071 条 `prompt/chosen/rejected/chosen_score/rejected_score/requirements`。公开 HF 行不含独立的逐 criterion score 列；若下载 RLCF 仓库另行链接的预计算 scoring 中间产物，必须单独记录 revision/checksum。
2. 在统一 trainer 中运行 released-data 上的 `RLCF pairs + standard DPO` baseline；这是 released-data reuse/baseline run，不称原 recipe 或完整 checklist generation/scoring pipeline reproduction，也不要把 RLCF 写成新的 margin loss。
3. 直接定义 `m_raw_i=chosen_score_i-rejected_score_i`，再按第 2.1 节只用 train split 冻结 `q95_WC`。只有在另行取得并验证 criterion-level 中间产物后，才做逐 criterion 加权重构检查。ROIV 仍需从 `requirements` 解析 criteria/importance，并在共同 calibration response bank 上用 common scorer 重新产生 criterion-score matrix。
4. 使用与 UltraFeedback 相同的矩阵：DPO、MMPO nominal/preference-label-uniform/ours，以及 ODPO/Scaled-DPO normalized-gap transfer nominal/loss-DRO-uniform/ours。
5. 若要重新生成 checklist，只在预计算数据复现通过后进行。

### 8.3 HelpSteer2

1. 使用官方 ratings split 中同 prompt 的两条 responses，保留 helpfulness、correctness、coherence、complexity、verbosity 五项完整评分。
2. 主聚合写为 `0.65*helpfulness + 0.80*correctness + 0.45*coherence + 0.55*complexity + 0.40*(4-verbosity)`；它与 NVIDIA 的 `-0.40*verbosity` 只相差常数，pair gap 完全相同。显式取 `chosen=argmax r`、`rejected=argmin r`、`m_raw=r_chosen-r_rejected`，再按第 2.1 节用 train split 冻结 normalization。另做只含前三项的 goodness sensitivity。不得把五维原值同向相加，也不得用独立 preference strength选择 pair 或改写 `m_i`。
3. 使用 HelpSteer2-Preference 最终发布的 7,118 对（6,766 train / 352 validation）及有符号 `preference_strength in {-3,-2,-1,1,2,3}` 只做 held-out validity 和 margin-calibration diagnostic；原始 `-100` 与聚合为 0 的样本已从论文最终数据中排除。
4. 跑 DPO、MMPO nominal/preference-label-uniform/ours，以及 ODPO/Scaled-DPO normalized-gap transfer nominal/loss-DRO-uniform/ours。HelpSteer2-Preference 的 native strength `{1,2,3}` Scaled DPO 另作 source-formula parity/可选独立 reproduction lane；没有独立官方 trainer，因此不能称 official-code reproduction。

### 8.4 Native/structured-feedback 主表形状

| 方法 | UltraFeedback | WildChecklists | HelpSteer2 |
|---|---:|---:|---:|
| Base checkpoint | yes | yes | yes |
| Vanilla DPO | yes | yes | yes |
| MMPO nominal / uniform / ours | yes | yes | yes |
| ODPO nominal / loss-DRO uniform / loss-DRO ours | yes | yes | yes |
| Scaled-DPO normalized-gap transfer nominal / loss-DRO uniform / loss-DRO ours | yes | yes | yes |

截图中的 nominal 与 ours 之间必须补 `uniform robust (s=1)`。没有这一行，无法排除收益只是一般 minimax robustness，而非 rubric-conditioned `s`。MMPO 的 uniform/ours 是 preference-label DRO；ODPO / Scaled-DPO normalized-gap transfer 的 uniform/ours 是第 2.5 节的 loss-DRO，表脚必须明确。

### 8.5 Seed promotion

- 所有行先跑 1 个固定 seed，排除实现错误和明显失败配置。
- 三套 native/structured-feedback systems 的 DPO 与 MMPO nominal/uniform/ours 最终跑 3 seeds；ODPO / Scaled-DPO normalized-gap transfer extensions 先全矩阵单 seed，预注册代表性 systems 再升 3 seeds。若把其余单元放入主表，则必须一并升到相同 seed 预算。
- ML-RDPO 附录至少 1 seed；若成为强竞争者再升到 3 seeds。

**Gate 2：** Vanilla DPO 与原 recipe 趋势合理；所有 nominal baselines 可复现；ours 相比 uniform 的差异不是由长度、policy KL、训练不稳定或调参预算造成；HelpSteer2 上 rubric-derived direction 与独立 human preference direction 至少显著优于随机，否则先审计 aggregation。

该阶段中 UltraFeedback 与 HelpSteer2 是全局固定 rubric，主要验证自动 robust-strength calibration和 sidecar transfer；WildChecklists 的 prompt-specific requirements 已经产生 rubric-level variation，可以同时提供单数据内的 geometry 证据。

## 9. Phase 3：无原生 rubric 数据上的 automatic-rubric 主实验

### 9.1 截图表格的正确含义

主实验确实按**完整 rubric system** 分列，而不再按论文逐个建立互不相干的数据集表。列分为两类：

- native/structured-feedback systems：UltraFeedback、WildChecklists、HelpSteer2 structured-feedback adaptation；
- automatic rubric systems：Auto-Rubric、OpenRubrics、Rubric-ARM、OnlineRubrics-local、RRD-local、frozen EvoLM。

因此截图还应增加 `HelpSteer2` 一列，并把表头标题写成 `Rubric system / source`，不能把所有列统称为 dataset。前三列使用各自发布的 native/structured-feedback data；automatic 六列的主政策比较使用同一 common WildChat bank。不同 native/structured-feedback 数据列与 method-native released-data 表之间**只做列内比较**，主要报告 `ours - uniform`，不能根据绝对 policy score 给 rubric producer 排名。

### 9.2 每个 system 的精确数据与实现标签

| 主表列 | 原论文/官方数据事实 | 本文可执行数据 | 正确标签 |
|---|---|---|---|
| UltraFeedback | raw 4-aspect scores + cleaned binarized pairs | 对齐后的共同 pair set | UltraFeedback native rubric |
| WildChecklists | 论文流程计算逐 criterion 分数并按 importance 聚合；官方 HF 最终 release 为 51,071 行，只公开 `requirements`、pair 与 aggregate `chosen_score/rejected_score`，没有 criterion-score 列 | 首轮复用 released pairs，并令 `m_raw=chosen_score-rejected_score` 后按统一规则归一化；完整 scoring pipeline 另行 gated | WildChecklists native released-data reuse |
| HelpSteer2 | ratings 有 21,362 条五维 0--4 分；Preference auxiliary file 最终有 7,118 对独立人工方向/强度，但没有逐行自然语言 rubric 字段 | 使用论文完整标注指南与 model-card-derived scalarization adapter 构 `m_raw`，再按 §2.1 的 train-only q95 得 `m`；人工 ±1/±2/±3 strength 只做外部效度 | HelpSteer2 structured-feedback adaptation / human-magnitude anchor |
| Auto-Rubric | 官方 HF 现有 38,459 条均为 HelpSteer3-derived query-specific annotations；论文还从 HelpSteer3/UltraFeedback 学 global Theme--Tips，并在 WildChat 做 DPO | released-data artifact check；common WildChat 主表只用论文 Appendix K 的 HelpSteer3-source **global** rubric，不把 query-specific rubric 跨 prompt 套用 | published HS3 global-rubric reuse + common-scorer adaptation |
| OpenRubrics | OpenRubric-v2 74,214 rows，含 pair、winner、rubric、judge trajectory；官方有 generator/judge code 与 4B/8B weights | official generator 在 common WildChat 产 rubric；common scorer 构 score/m | OpenRubrics generator + common scorer |
| Rubric-ARM | generator/judge 训练在 OpenRubrics general-domain portion；无独立 ARM preference dataset；官方发布 code/8B weights | frozen official generator 在 common WildChat 产 rubric；不重训 alternating GRPO | Rubric-ARM generator + common scorer |
| OnlineRubrics | 原 Generalist/Expert human-rubric datasets 与代码未公开；原方法使用 current/control rollouts、o3-mini extractor、GPT-4.1-mini grader 与 GRPO | 独立 induction split 只用于产生并冻结 current/control policy snapshots；随后在每个 common prompt 上生成 seed `C0(x)`，再从两套 policy 的 8 对辅助 contrast rollouts 生成 `Ce(x)`，冻结 `dedup(C0 union Ce)` 后用 common scorer 给共享 candidate bank 评分 | OnlineRubrics-local frozen-policy elicitation (DPO adaptation) |
| RRD | 原训练是 4k English non-toxic de-duplicated WildChat + Dr.GRPO；原作者未公开代码或精确 4k IDs | 同条件 common WildChat；复用 EvoLM 作者仓的 RRD-WU 后续实现；冻结 Qwen2.5-32B proposer、Qwen3-1.7B filter judge 与独立 strong/weak references；用 RubricARROW `1[p_true>=p_false]` 二值 matrix 重算 WU | RRD-local/common-binarized-WU adapter (EvoLM authors' reimplementation) |
| EvoLM | 原训练 prompts 来自约 271k Tulu-3 preference mixture；有官方完整 code 与 EvoLM-8B weights | frozen official checkpoint 在 common WildChat 生成 rubric；这是 in-distribution artifact reuse，不是完整 co-evolution 复现 | frozen EvoLM + common scorer |

不能把 UltraFeedback/HelpSteer2 上的 OnlineRubrics、RRD 或 EvoLM 行写成“原论文复现”：对 OnlineRubrics 它们是 local adaptation；对 RRD 是 transfer；对 frozen EvoLM，UltraFeedback/WildChat 最多是官方 checkpoint 的 in-distribution artifact reuse，HelpSteer2 是 transfer。

### 9.3 Automatic systems 的 common data generation

- 固定 English、non-toxic、去重后的 WildChat `4k train / 500 validation`，另加不重叠的 `500 induction` 专供 OnlineRubrics-local；WildChecklists 另留出 500 prompts 作跨方法 calibration diagnostic。所有 splits 与最终评测去重。精确 IDs/hash 固定后不得替换；若 pilot 后扩容，必须在任何正式训练前统一预注册。
- 对上述每个 prompt 先生成共享 reference answer：冻结 Qwen2.5-32B-Instruct、temperature 0；它只供 Prometheus scorer sensitivity 使用，并复用为 RRD strong reference。另用冻结 Qwen2.5-1.5B-Instruct 生成 RRD weak reference。两种 references 与 common candidate/Online auxiliary banks 全部隔离。
- 用冻结 `Qwen/Qwen3-8B`（`enable_thinking=false`）对 common train/validation prompt 各采样全局冻结的 `M` 个 responses（主配置 16）；Auto-Rubric global、OpenRubrics、Rubric-ARM、RRD-local、frozen EvoLM 复用同一 response bank。每个 prompt-specific rubric 的 `s(C_i)` 在 pair top/bottom 选择前，使用该 prompt 的全部 `M` 个 criterion-score vectors 计算；不使用最终 chosen/rejected 标签作为 `s` 的输入。
- 对 OnlineRubrics induction split 单独用同一冻结 `Qwen/Qwen3-8B`（`enable_thinking=false`）采样 `M_induction=8`（temperature 0.7, top-p 0.9, 与 common bank 相同 max-new-tokens），使用独立 seed 并保存独立 hash；它只供 induction-only pair/warm-up，不进 common candidate bank。
- OnlineRubrics-local 的离线适配冻结为以下四步，不能把 induction prompts 的 rubric 文本跨 prompt 套用：
  1. 在独立 500-prompt induction split 上，用论文 synthetic-offline-rubric prompt 与冻结 `Qwen/Qwen2.5-32B-Instruct` extractor（exact revision、temperature 0）生成 prompt-specific seed criteria `C0(x)`；
  2. 用 `C0`、common scorer 和 induction-only candidate responses 构 pair，做一轮预注册 nominal-MMPO warm-up，得到 `Pi_current`；`Pi_control` 始终是初始冻结 revision 的 `Qwen/Qwen3-8B`（`enable_thinking=false`）；
  3. 冻结 `Pi_current`、`Pi_control`、extractor、decoding、warm-up steps 与 dedup 配置；
  4. 对 common train/validation/calibration 中的**每个 prompt**，先用同一 synthetic prompt 独立生成 `C0(x)`；再由 current/control 各采样 8 个 rollouts，按相同 sample index 构 8 对 contrast，提取 `Ce(x)`；最终 rubric 明确定义为 `C(x)=dedup(C0(x) union Ce(x))`。把 `C(x)` 交给 common scorer，只对所有 producers 共享的 candidate response bank 形成 score/pair/m。辅助 rollouts 只用于 rubric elicitation，不进入 candidate comparison。所有 prompt-specific rubrics 在任何 consumer 训练前冻结。
- 这一路径没有使用 common validation/evaluation labels，但仍是本文明确规定的 local DPO adaptation，不是原论文数据/GRPO/API setting 的 reproduction。
- 六套 rubric 文本都交给同一个冻结 `OpenRubrics/RubricARROW-8B-Judge` reference-free pointwise criterion grader；严格按第 2.3 节用官方 `p_true-p_false` 规则构造 `[0,1]` criterion score、使用同一 parser/failure contract，且不删除 prompt。这张表测的是 `rubric generator/text + common-scorer adapter`，不是各方法完整 native judge system。Prometheus 只在预先固定的同一 200-prompt 子集作 scorer sensitivity，不能改变主 pair、`m` 或 `s`。
- RRD main controlled row 保留 recursive proposal/filter，但 strong/weak references 均是冻结且与 candidate 分离的 banks；使用 RubricARROW `1[p_true>=p_false]` 形成每个 recursive rubric 的 binary matrix，再按 `rrd_wu` 重算 weights，并形成 aggregate score/m 与 rubric-level `s_RRD^common(C_i)`。RRD native Qwen3-1.7B judge 只作 sensitivity并另存 rubric-level `s_RRD^native(C_i)`，不得混用。
- 按 producer 发布的 importance/aggregation rule 得到 aggregate score；没有权重时使用等权并记录为 adapter choice。先 canonicalize criterion set 并缓存 `rubric_id -> s(C)`，再在每个 prompt 取最高/最低 aggregate score为 chosen/rejected，定义 `m_raw=score_max-score_min`；raw ties 丢弃。
- 每个 producer 严格按第 2.1 节，只在自己的 train split 冻结 `q95_g`，得到 clipped normalized `m` 并丢弃 `m<0.05` near-ties；validation/test 复用 train `q95_g`。缓存原始 rubric、criterion scores、aggregate score、winner/loser、raw/normalized `m`、judge output 和 parsing status。
- 原方法 native pairwise judge 可以另做 sensitivity。若用 repeated pairwise judging 而非 pointwise scorer，则两种顺序各重复 `K` 次并拟合 Bradley--Terry utility；所得量只能称为 estimated/confidence-derived margin，不能称为人工真实 gap。

### 9.4 主 coverage matrix 与优先级

最终 coverage matrix 的列就是第 9.2 节九个 systems，行应为：

| 方法行 | MMPO 接口 | ODPO / Scaled-DPO normalized-gap transfer 接口 | 优先级 |
|---|---|---|---|
| Base checkpoint | n/a | n/a | P0 |
| producer-specific Vanilla DPO | 忽略同一 pair 的 `m` | 同左 | P0 |
| nominal margin consumer | original or stabilized MMPO（取决于 clipping） | original ODPO / pre-registered Scaled-DPO normalized-gap transfer | P0 |
| `+ uniform robust (s=1)` | preference-label DRO | baseline-preserving loss-DRO | P0 for MMPO；P1 for ODPO / adapter |
| `+ ours (s=s(C))` | preference-label DRO | baseline-preserving loss-DRO | P0 for MMPO；P1 for ODPO / adapter |

截图需要四项必要修正：表头明确写 `Rubric system / source`；加入 HelpSteer2 structured-feedback 列；在 nominal 与 ours 之间加入 uniform robust 行；将 `Scaled DPO` 改名为本项目的 `Scaled-DPO normalized-gap transfer`，不声称已有论文方法。每个 system 的 DPO 使用该 system 自己选出的 pair，但忽略 `m`；同一 system 内 nominal/uniform/ours 必须使用完全相同的 pairs、`m`、initialization 和预算。

执行与报告分三层，避免截图中的“全覆盖”与统计主表混淆；层级不绑定具体 GPU 型号或卡数：

1. **九-system MMPO 主表（3 seeds）：** Base、producer-specific DPO、MMPO nominal、MMPO+uniform、MMPO+ours；这是论文的核心表。
2. **consumer-transfer 表（3 seeds）：** ODPO / Scaled-DPO normalized-gap transfer 只在预注册的 UltraFeedback、WildChecklists、Rubric-ARM、frozen EvoLM 四个代表 systems 上报告 nominal/uniform/ours。
3. **coverage appendix（单 seed）：** ODPO / Scaled-DPO normalized-gap transfer 在其余五个 systems 的完整可插拔性结果。

因此截图仍可作为“需要运行的 coverage matrix”，但不能把单-seed ODPO / Scaled-DPO normalized-gap transfer 单元与三-seed MMPO 主结果混在同一统计主表。若论文最终要把两个 transfer consumers 的全九列放主表，则必须全部升到相同 3-seed 预算，不能只提升结果较好的单元。

### 9.5 自然多-rubric 联合训练：核心识别实验

取消 1D/2D/3D 后，本阶段承担 `s(C)` 的核心识别：

1. 使用同一批 WildChat prompts、同一批冻结的 `M` 个 responses 和同一个 common grader。
2. 六个 producers 对 prompt `i` 生成 rubric `C_{gi}`。Auto-Rubric-global 的所有 `C_{gi}` 相同；其他五个主轨通常为 prompt-specific，逐 exact criterion set 得到 `s(C_{gi})`。common grader 是共享冻结测量工具，不属于 rubric。
3. 每个 system 等量采样训练 pairs，固定 `pi_g=1/6`；所有对照使用完全相同的 multi-system mixture 和总训练步数。
4. 在一个联合 ambiguity set 中训练：

\[
\sum_g\frac{\pi_g}{n_g(-\log\varepsilon)}\sum_i
s(\mathcal C_{gi})D_{\rm KL}(Q_{gi}\Vert P^0_{gi})\leq\rho.
\]

5. 比较：

| Joint setting | 作用 |
|---|---|
| MMPO nominal mixture | 无 robust layer |
| MMPO + uniform robust | 所有 rubrics 均用 `s(C)=1` |
| MMPO + ours | 每个 exact rubric 使用自己的 ROIV `s(C)` |
| MMPO + within-producer shuffled-s | 在每个 producer 内打乱 `rubric_id -> s` 映射，保持该 producer 的权重 multiset 不变；global rubric 无可打乱项 |
| MMPO + within-producer rank-reversed-s | 在每个 producer 内按 rubric-level `s` 排序后反向赋值，保持 multiset/均值不变，不用 `1/s` |
| MMPO + cross-producer shuffled-s | 仅作附录：在匹配样本数后跨 producers 打乱完整 rubric-level 权重 multiset |

shuffling 的单位必须是 `rubric_id`，不是裸样本行：如果多个样本共享同一 rubric，它们在错配后仍共享同一个被分配的 `s`。prompt-specific rubric 恰好一 prompt 一 rubric 时，rubric-level permutation 在数据行上看似 sample shuffle，但其语义仍是打乱 rubric-to-s mapping。

### 9.6 Method-native artifact checks（分表，不横向排名）

| 方法 | 可公开复现的 native check | 不能声称的内容 |
|---|---|---|
| Auto-Rubric | official HelpSteer3-derived 38,459-row data；按 normalized input 分组切 train/val | 不能说公开 release 同时含 UltraFeedback，也不能说有连续 `m` 或生成 checkpoint |
| OpenRubrics | OpenRubric-v2 + official RubricRM-v2 generator/judge | binary winner/judge trajectory 不是连续 `m` |
| Rubric-ARM | official checkpoints 在 OpenRubrics general-domain held-out slice 做 sanity；再按官方 dual-order rule 造 pair | 没有独立发布的 ARM preference dataset 或 final DPO pair dump |
| OnlineRubrics | 只能核对论文 prompts/statistics | Generalist/Expert 数据未公开，不能称 faithful reproduction |
| RRD | EvoLM authors' RRD-WU reimplementation 在 WildChat 近似复现 | 不是 RRD original-author code；精确 4k prompt IDs 未公开 |
| EvoLM | official code/weights + Tulu-3 provenance；使用 frozen producer | frozen inference 不是完整 paper co-evolution 或 full-training reproduction |

这些 method-native checks 回答“公开产物是否可用”，不用于给 producer 做绝对优劣排名。Rubric-ARM 是 producer/system 列，不是一个新的独立 dataset；OpenRubric-v2 不能同时算两份独立训练证据。

### 9.7 API 决策

主 controlled track 不调用商业 API：candidate responses、rubric generation、criterion grading均在本地完成。只有以下 faithful sensitivity 需要 API，且不阻塞主实验：

- OpenRubrics 原始 frontier-model 数据重造；
- OnlineRubrics 原文 o3-mini extractor + GPT-4.1-mini grader；
- RRD 原文 GPT-4o proposer。

预算允许时，每种仅抽取 200 prompts 做 `local vs faithful` rubric/pair agreement，不用 API 生成全量 DPO data。

**Gate 3：** rubric producer 之间除 rubric 本身外，共享 prompts、responses、RubricARROW revision、parser、pair rule、margin normalization、training budget 和 evaluation；joint mixture 中每个 system 样本数相同。每个 exact rubric 都在其冻结 assignment prompt 的共同 response bank 上估计 `s(C)`；同一 rubric hash 必须共享缓存，不按 producer 求一个平均 `s`。每个 producer 的 criterion parse success 必须 `>=99%`，train-only `q95_g>0`，有效 non-tie/near-tie 后 pair 数足以满足冻结样本数，且 common-grader score direction 在独立 preference diagnostic 上显著优于随机；否则只将该 producer降级为 artifact check/local adaptation report。六-producer policy table 可取共同有效的 WildChat prompt IDs，但该交集不回流改写已经计算的 rubric-level `s(C)`。

## 10. Phase 4：online/evolving rubrics（扩展实验）

offline 主实验成立后再做 OnlineRubrics-local 的 2--3 轮 IterDPO：

1. 从相同初始 checkpoint 开始。
2. fixed control/reference policy 与 current policy 在同一个 training prompt pool 上分别采样；保存 policy id，不能把同一 policy 的八个回答伪装成 current/control comparison。
3. OnlineRubrics-local 按 pairwise contrasts 提取、去重并更新 prompt-specific mapping `C_t={C_t(x): x in D}`，其中每个 `C_t(x)=dedup(C_0(x) union C_{e,t}(x))`；原论文使用未公开数据和闭源 extractor/grader，本地版必须明确标记 adaptation。
4. common grader 在冻结 response bank 上生成 `m_i^(t)`。
5. 每轮对 training/calibration 中每个 prompt 的 `C_t(x)` 分别计算 `s_t(C_t(x))`；exact rubric 重复时复用缓存。不能把一个 prompt 的 rubric 搬到另一个 prompt，也不能把一轮的整套 mapping 压成一个 system-level scalar。
6. 用 MMPO 更新一轮 policy，再进入下一轮；control policy 与全局 `rho` 始终不变。

四个控制：固定初始 rubric、只更新 rubric、online rubric + uniform robust、online rubric + ours。所有轮次使用同一个 `rho`。原论文使用 GRPO，因此这里统一标注为 DPO adaptation。EvoLM 的完整 co-evolution 和 Rubric-ARM 的 alternating-GRPO 不进入主阻塞路径；主表只使用它们的冻结 producer。

## 11. Phase 5：必须完成的 ablations

### 11.1 `s` ablations

- `s=1`：最重要的 uniform robust control。
- criterion count proxy。
- rubric text-embedding log-det proxy。
- ROIV 主方法。
- within-producer rubric-level shuffled `s(C)`：打乱 `rubric_id -> s` 映射；相同 rubric 的重复样本保持绑定。
- within-producer rubric-level rank-reversed `s(C)`：反向赋值同一组 rubric weights，不使用 `1/s`，保持 multiset/均值。
- global-rubric systems 因只有一个 `s`，不伪造 within-system shuffle；只参加跨-system附录控制。
- ROIV with/without judge-noise subtraction。
- calibration size `L/M` 敏感性。
- `lambda in {0.1,1,10}` 的全局敏感性。

### 11.2 Robustness ablations

- `rho` 的全局敏感性：每次对所有 rubrics 共同采用一个常数，绝不使用 rubric-specific `rho`。
- KL 主结果；chi-square 作为附录距离。
- common grader 与 method-native grader 的小规模敏感性。
- probe response seed 与备用 probe policy 的小规模排序稳定性。

## 12. 评测顺序与指标

### 12.1 先做与论文主张直接相关的评测

- held-out preference NLL、accuracy/AUC、calibration error；
- rubric-derived score margin 与独立 judge preference rate；
- 全局 rubric 的 `s`/CI/effective radius；prompt-specific producer 的 rubric-level `s(C_i)` 分布、分位数、stability 与 adversarial shift-by-`s`；
- MMPO inner label shift `mean |q_i^*-p_i^0|`；ODPO / Scaled-DPO normalized-gap transfer extensions 另报 empirical sample-weight shift / effective sample size；
- policy-reference KL、DPO reward margin、response length、gradient norm、dual variable；
- judge call 数、rubric generation cost、training GPU hours。

### 12.2 Specialized policy benchmarks

- AlpacaEval 2 length-controlled；
- Arena-Hard style-controlled；
- WildBench；
- IFEval、InfoBench；
- WildChecklists/RLCF 轨道增加 FollowBench。

### 12.3 最后运行五个 EvalScope 通用 benchmarks

- MMLU；
- GSM8K；
- BBH；
- HumanEval；
- GPQA-Diamond。

通用 benchmark 不用于挑选 checkpoint。先由 validation 与 specialized metrics 冻结 checkpoint，再统一评测通用能力和 regression。

## 13. 统计检验与主张判定

- 主表报告 3-seed mean、standard deviation 和 95% CI。
- 同一 evaluation prompts 上使用 paired bootstrap 比较 ours 与 uniform robust。
- judge-based evaluation 做 response-order swap，报告 position-bias sensitivity。
- 同时报告平均 response length，避免把长度增加误认为质量提升。
- 多个 benchmark 的结果同时报告，不只汇报正向子集。

核心主张的预注册判定：

1. **rubric-conditioned 主识别：** natural multi-system joint training 中，ours 优于 uniform、shuffled-s 和 rank-reversed-s；ours-vs-uniform 的 paired-bootstrap 95% CI 不跨 0。
2. **单系统/单数据自动校准：** UltraFeedback、HelpSteer2 的全局 rubric 结果与 validation-tuned scalar-radius control 具有竞争力；WildChecklists 的 prompt-specific requirements 上，ours 相对 fixed-`rho` uniform 为正，且 rubric-level shuffled/rank-reversed controls 不能取得同等收益。
3. **consumer 泛化：** MMPO preference-label DRO 的 `ours - uniform` 为正；ODPO、Scaled-DPO normalized-gap transfer 的 baseline-preserving loss-DRO extensions 中至少一个同方向，且没有明显能力/长度 regression。后两者不能冒充对 MMPO label-ambiguity objective 的独立复现，adapter 也不能冒充 HelpSteer2 paper 的 DPO reproduction。
4. **producer 泛化：** 六个 automatic-rubric producers 中至少四个的平均 `ours - uniform` 为正；同时分别报告“官方代码/权重”三者和“发布资源/本地适配”三者，不能只汇总掩盖 provenance 差异。
5. **s 的可信度：** `s` 的估计稳定，且与独立 measurement validity 的关系不与理论方向系统性相反。

若第 1 条不成立，不能把方法写成有效的 rubric-conditioned geometry；最多只能报告单系统自动半径校准。若第 5 条不成立，应把贡献降级为一种 heuristic distance weighting，而不是 rubric sufficiency measurement。

## 14. 缓存、目录与可复现产物

推荐目录：

```text
configs/
  data/
  calibration/
  train/
  eval/
data_manifests/
cache/
  generations/
  rubric_outputs/
  criterion_scores/
  margins/
  reference_logps/
artifacts/calibration/<system_id>/
artifacts/experiment/<run_id>/
```

每个正式 run 必须保存：

- resolved config；
- dataset/model/code revision；
- seed；
- environment 与 GPU snapshot；
- command 与完整 log；
- `rubric_system_id`、canonical `rubric_id`、per-rubric `s(C)`、`rho`；
- checkpoint；
- metrics JSON/Markdown；
- 与对应 nominal/uniform baseline 的 delta；
- 成功、失败或不可比较的结论。

## 15. 资源自适应的执行顺序与预算控制

1. 冻结 environment、repos、data revisions 与全部 prompt splits；首先从 WildChecklists 留出并 hash 固定 500-prompt common calibration bank。
2. baseline loss/source-parity 单测、UltraFeedback 1k DPO/MMPO/ODPO/Scaled-DPO smoke、reference cache 和 pilots 已完成，不重复。Robust `rho==0`、`s=1`、dual/explicit-inner 测试保留到第 5 步，不与 baseline 训练交叉改代码。
3. 先完成 UltraFeedback seed-42 四 baseline 及 official-test 本地评测；当前 DPO 首次正式运行在 step-283 FSDP checkpoint save 处失败，因无 trusted checkpoint 须在 checkpoint 通路修复后从 step 0 重启。不重做 data/cache/smoke/pilot。
4. baseline 本地结果冻结后，先生成 UltraFeedback 500-prompt×`M=16` calibration bank，用冻结 RubricARROW scorer 计算 `s_UF` 与 bootstrap CI。
5. 随后才实现 KL 与 Wasserstein-W1 robust backends；W1 ground cost、rho normalization、solver 和选择规则必须先写回方法合同。通过 `rho=0`、`s=1` 和 explicit-inner gates 后，在 UltraFeedback development split 上仅用 uniform control 选择一次全局 `rho`。
6. 依次生成 shared Qwen2.5-32B reference bank、RRD Qwen2.5-1.5B weak bank、WildChat common response bank、独立 `M_induction=8` Online induction bank，以及需要的 native/probe response banks。所有 banks 分离并保存 hash；正式 `M` 在 pilot 后全局冻结。
7. 对 HelpSteer2 global rubric 在共同 500-prompt calibration bank 上计算一个 `s`；对 WildChecklists 实际进入训练/验证的每个 exact requirements checklist，在其来源 prompt 的 probe responses 上计算 `s(C_i)`。共同 500 prompts 只用于跨方法 diagnostic，不再生成一个错误的 `s_WC` 常数。
8. 在独立 induction split 上按第 9.3 节生成 `C0(x)`，用 induction-only candidate bank 完成一轮 nominal-MMPO warm-up，并冻结 OnlineRubrics-local 的 current/control policies、Qwen2.5-32B extractor、decoding 与 dedup 配置；不得使用 common validation/evaluation labels。
9. 在 WildChecklists 500-prompt diagnostic bank 上生成六个 automatic producers 的 rubrics并检查 `s(C)` 分布/parser/scorer；Online 使用逐 prompt `dedup(C0(x) union Ce(x))`，RRD 使用冻结 references、本地 actors 与 common-binarized-WU。
10. 在 WildChat common train/validation 上生成并冻结六套 prompt-specific/global rubrics；canonicalize 后用共同 response bank 同时计算每个 `s(C_i)`、criterion scores、pairs 和 margins。只有 exact global/repeated rubric 才共享缓存。完成 Gate 1/3 后，所有数据与 rubric-level `s(C)` 才可用于训练。
11. UltraFeedback、WildChecklists、HelpSteer2 全矩阵单 seed，并用独立 human signals 完成 validity gate；1k smoke 已在第 2 步完成，不重复。
12. 只把通过 gate 的 native/structured-feedback DPO/MMPO 主表行提升到 3 seeds。
13. 六个 automatic-rubric 的 producer-specific DPO 与 MMPO nominal/uniform/ours sweep 单 seed。
14. 六-system joint MMPO：nominal/uniform/ours/shuffled/rank-reversed。
15. ODPO / Scaled-DPO normalized-gap transfer 的 loss-DRO extensions 先在九个 systems 跑单 seed；预注册 UltraFeedback、WildChecklists、Rubric-ARM、frozen EvoLM 升 3 seeds。
16. 九-system 的 DPO/MMPO nominal/uniform/ours 主表与 joint table 提升到 3 seeds；ODPO / Scaled-DPO normalized-gap transfer 只将预注册四-system transfer 表提升到 3 seeds，其余保留单-seed appendix。
17. 完成 ablations、JudgmentBench premise diagnostic、Prometheus-vs-reference-free scorer sensitivity 与 specialized evaluation。
18. 资源允许且 offline 结论成立时运行 OnlineRubrics-local 2–3 round extension；faithful API 只做 200-prompt sensitivity。
19. 冻结 checkpoints，最后运行五个通用 benchmarks。

安全的节省方式：缓存数据生成与 reference log-prob、FSDP full-shard/ZeRO-3、BF16、FlashAttention、gradient checkpointing、单 seed 筛选后再升 3 seeds。LoRA 仅用于 smoke 或单独的参数高效 ablation。不能通过改变数据量、epoch、evaluation 或只给 ours 更多调参预算来节省成本。

## 16. 停止与回退条件

- baseline 无法复现：停止 ours 全量训练，先修 baseline/data contract。
- `s` 对 calibration resampling 极不稳定：增加 calibration 数据或修评分解析。
- uniform 与 ours 在 `s=1` 时不相等：视为实现错误。
- ours 的改进完全由更大 policy KL 或 response length 解释：不能作为方法收益。
- automatic-rubric producer 的 prompt/response bank 不同：重新生成 common pool，不做不公平横比；producer 选出的 pair 可以不同，但比较必须使用共同有效 prompt intersection 和相同样本数。
- automatic-rubric parser success、non-tie rate 或 common-grader validity 未过 Gate 3：该 producer 降级到 method-native/appendix，不强行进入 joint mixture。
- OnlineRubrics/RRD 使用本地替代模型时没有清楚标注 adaptation/provenance：结果不可报告为原方法复现。
- 两次完整且可比的主要重试仍显示 ours 系统性弱于 uniform：停止扩展实验，回到方法假设。

## 17. 修订记录

| 日期 | 修改 | 原因 | 对可比性的影响 |
|---|---|---|---|
| 2026-08-10 | 删除 1D/2D/3D 与人工 rubric omission 设计；改为完整 rubric systems 的 ROIV、稳定性和独立效度验证 | 用户明确要求不做人为删维度 | 减少人工构造，但将“真实 completeness”结论限制为 operational sufficiency proxy |
| 2026-08-10 | 完成 DPO/六个 rubric producers 的 code-data-API 审计；加入 HelpSteer2 与 JudgmentBench；automatic track 改成六 producers 的共同 WildChat response bank | 用户要求核查开源性、DPO 可改造性与 API 需求 | 用 common grader 隔离 rubric 差异；明确官方复现、本地适配和后续复现的 provenance |
| 2026-08-10 | final matrix audit：按 rubric system 分列；补 HelpSteer2 与 uniform rows；纠正 Auto-Rubric/OnlineRubrics/RRD/EvoLM 数据标签；区分 MMPO label-DRO 与 ODPO / Scaled-DPO normalized-gap transfer loss-DRO extensions | 用户要求按 producer/system 组织主实验并保证方法--数据映射准确 | 避免把 transfer/adaptation 写成复现，且保证每个 `+ours` 在 `rho=0` 有合法 nominal baseline |
| 2026-08-10 | final consistency pass：统一 prompt 内 ROIV 统计口径；将 signed rubric scores 改写为 direction-normalized positive importance；两类 KL 按冻结上界无量纲化但保留一个固定 `rho`；明确 OnlineRubrics-local 逐 prompt elicitation 与三层 seed/reporting | 独立数学、数据与 provenance 复核发现固定/prompt-specific rubric 不可比及 consumer 接口尺度问题 | 当时采用 system-level `s_g`；该粒度已由下方 2026-08-11 rubric-instance correction 取代，保留本行仅作历史记录 |
| 2026-08-11 | final source/executability pass：预留 WildChecklists 500 prompts；冻结 reference-free RubricARROW 主 scorer/parser、Prometheus sensitivity、RRD local actors与 Online induction bank；定义 rank-reversed-s；将 `rho` 冻结放入执行顺序 | 解决 prompt-specific requirements、reference 自比较门控、WU 二值输入与 preflight 依赖链 | scorer/parser/provenance 合同继续有效；“九个 system-level `s_g`”口径已由下一行取代 |
| 2026-08-11 | rubric-instance correction：rubric 明确定义为 criterion set，不含 prompt/producer；global rubric 共享一个 `s`，prompt-specific rubric 逐 exact criterion set 计算并缓存；加权 KL 改为逐 rubric；删除单张 H100 硬件合同 | 用户指出 automatic producers 通常逐 prompt 生成 rubric，不能压成一个 system-level `s` | WildChecklists/OpenRubrics/ARM/Online/RRD/EvoLM 现在使用 rubric-level `s(C_i)`；相同 rubric 共享值，资源配置运行时记录而非预先写死 |
| 2026-08-11 | 将主表训练从 LoRA 更正为 `Qwen/Qwen3-8B` BF16 full-parameter fine-tuning，使用 FSDP full-shard/ZeRO-3 与 reference-logp cache；LoRA 降为 smoke/独立 ablation | 用户明确主模型为 Qwen3-8B；原始 DPO、ODPO、MMPO 7B/8B 与 RLCF 官方训练入口也均不是 LoRA-only | 所有 nominal/uniform/ours 行共享同一 Qwen3-8B revision、non-thinking chat-template 和全参训练合同；避免模型身份或 LoRA 容量约束混淆 rubric robustness 效果 |
| 2026-08-12 | 完成 NUS 现状审计；UltraFeedback smoke/pilots 可信，seed-42 DPO 在 step-283 checkpoint save 失败；执行顺序改为 baseline → `s_UF` → KL/W1 → 其他 native/automatic tracks | 正式 run 的 manifest 与真实进程冲突，且用户要求先完成 baseline、再计算 `s_UF`、最后接 robust 方法 | 不改 baseline 超参/数据/预算；失败 run 不计为结果，其他数据 raw assets 复用但必须另做 production gates |
