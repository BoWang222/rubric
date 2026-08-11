# OnlineRubrics / RRD / EvoLM 实现与 DPO 适配审计

> **状态：历史调研材料（非执行规范）。** 本文件保留当时的候选方案与术语，可能与最终设计不同。正式实验只以根目录 [`PLAN.md`](../../PLAN.md)、[`CHECKLIST.md`](../../CHECKLIST.md) 和 [`final_experiment_matrix_audit.md`](../scout/final_experiment_matrix_audit.md) 为准；尤其不得从本文沿用旧称 “Scaled DPO”、旧 scorer/gate 或旧数据映射。

> 审计日期：2026-08-10。仅使用论文、作者/机构项目页、作者模型仓和数据维护方仓库作为方法事实依据。

## 结论先行

| 方法 | 准确论文 | 原始优化 | 官方可用资源 | 是否直接产生可用分数 | 忠实复现需闭源 API | 单张 H100 80GB 判定 | 建议地位 |
|---|---|---|---|---|---|---|---|
| OnlineRubrics | *Online Rubrics Elicitation from Pairwise Comparisons*, arXiv:2510.07284 | GRPO | 论文+机构页；未发现官方方法代码或 Generalist/Expert Rubrics 数据下载 | 是。每项二元满足度+权重，加权后为标量 reward | 是：原文 o3-mini extractor + GPT-4.1-mini grader；评估还用 Gemini 2.5 Pro | 原始在 8×H100 训练，单卡不宜完整复现；但离线产生对比数据可行 | 作为“动态 rubric producer”对照，重写抽取+去重+打分管线 |
| RRD | *Rethinking Rubric Generation for Improving LLM Judge and Reward Modeling for Open-ended Tasks*, arXiv:2602.05125 | Dr.GRPO（仅 policy RFT）；rubric 生成器不训练 | RRD 原作者未发布官方代码/checkpoint；但 EvoLM 作者官方仓内置了论文对比用的后续复现 `rrd_uniform/rrd_llm/rrd_wu` | 是。每项 rubric predicate 为 0/1，WU/其他权重聚合成标量 reward | 原论文 faithful 设置需 GPT-4o proposer；EvoLM 复现框架支持 API proposer 或本地 proposer/judge | 原论文报告 8×H100；120B judge 不适合单卡常规运行。改用小型本地 judge 可离线构数据 | 优先复用 EvoLM 仓的 RRD-WU 实现；明确标为后续复现，非 RRD 原作者代码 |
| EvoLM | *EvoLM: Self-Evolving Language Models through Co-Evolved Discriminative Rubrics*, arXiv:2605.03871 | policy 和 rubric generator 交替 GRPO | 作者已公开完整训练仓 `stellalisy/EvoLM` 和 `stellalisy/EvoLM-8B` 权重；原数据 Tulu 3 mixture 由 AllenAI 公开 | 是。JSON rubric 包含 criteria、weight、0–1 scoring levels；judge 输出分项和总分 | 主方法本身不要求闭源 API；仓库也可选接 GPT-4.1/Claude/Gemini | 论文原实验报告 64×H100；README 另写明完整训练建议 8×H100。单卡适合冻结 producer 推理和 DPO 适配，不建议完整共演化 | 六个 auto-rubric 中的高优先级主对照；代码可作为 EvoLM/RRD/Rubric-ARM 统一 RL 实现参考 |

## 1. OnlineRubrics

### 来源与原始方法

- 论文：https://arxiv.org/abs/2510.07284
- 机构页：https://labs.scale.com/papers/onlinerubrics
- 方法在每个训练 step 对当前 policy 和 control/reference policy 各采样 `M` 个回答，用 LLM extractor 从回答对差异中抽取新 criteria，再去重并加入初始 rubric。
- 原文默认比较 8 对 rollouts，最后约得 8 个 elicited criteria。
- grader 对每个 criterion 输出 0/1，以权重归一化加和得到 `R(x,y,C)`。
- 原始策略优化为 GRPO；Qwen-2.5-7B-Instruct，16 rollouts/sample，3 epochs，8×H100。
- 原文模型：o3-mini 抽取和去重、GPT-4.1-mini 打分，Gemini 2.5 Pro 做部分胜率评估。
- 数据：Generalist Rubrics（1,500 train/487 eval）和 Expert Rubrics（1,864 train/332 eval）是人工 prompt-specific weighted binary rubrics；论文/项目页未给出公开下载。

### DPO 改造

1. 从公平的共享 prompt pool 取 `x`，用同一个固定 sampler 产生 `K>=4` 个候选回答。
2. 额外生成 current/control 响应对，按论文 prompt 抽取 criteria 并去重。若数据无初始 rubric，先做一次 prompt-only synthetic rubric，再用 pairwise online criteria 增补。
3. 用统一 grader 计算每个候选的权重分数 `R_k`。
4. `chosen=argmax R_k`，`rejected=argmin R_k`，`m=R_chosen-R_rejected`；过滤 tie 及低于预注册阈值的 pair。
5. 把全部生成的 rubric 集合当作一个 producer/system，在固定 calibration bank 上计算一个 `s_OnlineRubrics`，不对单样本计算 `s_i`。

### 判定

- **可改造，但不是开箱即用 baseline。** 需自行实现 extractor/dedup/grader orchestration。
- **忠实版需付费 API。** 替换成开源本地模型可避免 API，但论文中应写成 `OnlineRubrics-local` 而非原方法完整复现。
- 单卡最合理的路线是“离线冻结 rubric producer 生成数据 + DPO”，不是重现 online GRPO。

## 2. RRD

### 来源与原始方法

- 论文：https://arxiv.org/abs/2602.05125
- 初始提案：给定 prompt 和 8 个 sample responses，LLM 提出初始 rubrics。
- 递归分解：如果某 criterion 被超过 2 个 rollouts 满足，就把它分解成更细 criteria。
- 过滤：删除方向错位 criteria（原文以是否反而偏好 Llama3-8B 相对 GPT-4o 作 guardrail），并用 LLM 去除重复/重叠 criteria；拒绝累计 15 个后停止。
- 打分：`g_k(x,y) in {0,1}`，`R(x,y)=sum_k w_k g_k(x,y)`。主变体 RRD-WU 使用未标注 rubric-score covariance 做 whitened-uniform 权重。
- 原始 RFT：4K 英文、无毒、去重 WildChat prompts；Qwen3-4B/Llama3.1-8B；Dr.GRPO；8 rollouts；1,000 steps；8×H100；`verl`。
- 原文在 RFT 中使用 GPT-4o 提 rubric，GPT-OSS-120B 判定 criterion satisfaction。
- 评估数据：JudgeBench、PPE；策略评估 BiGGen Bench 和 HealthBench-Hard。

### DPO 改造

1. 对共享 WildChat prompt pool 的每个 `x` 采样 `K>=8` 个回答。
2. 依论文 Algorithm 1 实现 initial proposal、decomposition、misalignment/redundancy filter 和 early stop。
3. 用同一本地 judge 得到 criterion matrix `G[K,d]`，在 calibration split 估 covariance，计算 WU 权重并得到每回答标量 `R_k`。
4. `chosen=argmax R_k`，`rejected=argmin R_k`，`m=R_chosen-R_rejected`。
5. 将固定 proposer+decomposition+filter+judge+WU 聚合整体视为一个 rubric system，得到一个 `s_RRD`。

### 判定

- **RRD 原作者仍没有发布官方代码，但已有高价值的论文对比复现可直接复用。** EvoLM 作者仓 https://github.com/stellalisy/EvoLM 在 `scripts/configs/reward_mode/` 提供 `rrd_uniform.sh`、`rrd_llm.sh`、`rrd_wu.sh`，并在 `open_instruct/search_rewards/rubric_judge_rewards.py` 实现递归 RRD 打分与 WU 聚合。
- **这是 EvoLM 论文作者的 baseline reimplementation，不是 RRD 原作者 code release。** 论文中应同时引用 RRD 原论文和 EvoLM 代码仓，避免把 provenance 写错。
- **忠实 proposer 依赖 GPT-4o API。** GPT-OSS-120B 虽为开放权重，单卡 80GB 用常规 BF16 不可行；4-bit 加 KV cache 也很紧，不建议作为主计划。
- 实验上应提前预注册两种版本：`RRD-faithful`（GPT-4o proposer）或 `RRD-local`（固定 Qwen3-8B/32B proposer+judge），不可混合报告。

## 3. EvoLM

### 来源与原始方法

- 论文：https://arxiv.org/abs/2605.03871
- 作者官方代码：https://github.com/stellalisy/EvoLM
- 作者模型权重：https://huggingface.co/stellalisy/EvoLM-8B/tree/main
- Tulu 3 官方数据集合：https://huggingface.co/collections/allenai/tulu-3-datasets
- 约 271K 去重 prompts，来自 Tulu 3 preference mixture，包含 UltraFeedback、WildChat、instruction following、math、code、scientific literature 和 personas。
- Qwen3-8B 参数共享地扮演 policy 与 rubric generator，Qwen3-1.7B 为冻结 judge。
- rubric JSON 包含 5–10（训练后常约 3–4）个 criteria，每个有正权重和 0–1 离散 scoring levels，权重和约为 1。
- judge 对每项 criterion 输出分数，权重加和为 `[0,1]` 总分。
- policy 与 rubric generator 都用 GRPO，每 50 steps 交替；8 samples/prompt；500 policy steps。
- rubric generator 的自监督 preference pairs 来自：时间对比（新 checkpoint 对旧 checkpoint）、inferred-question 和 rubric-conditioned response 对比。
- 原训练为 64×H100，最大 16,384 response tokens，具有 56 个 vLLM engines。
- 官方仓 README 另写明“8×H100 recommended for full training”。这是仓库的推荐可运行配置，不应用它覆盖论文 Appendix 报告的 64×H100 原实验 compute；两者应分开报告。
- 仓库提供 `scripts/train_rubric_policy_joint.py`、`scripts/launch.sh`、`rubric_data_provider.py`、GRPO trainer、本地/API judge provider，以及 OLMES、RewardBench 2 和 JudgeBench 评估脚本。
- 同一仓还提供 RAR、RRD（uniform/LLM/WU）、RLCER 和 Rubric-ARM 的 baseline reward modes。仓内未发现 `OnlineRubrics`/online pairwise elicitation reward mode，所以 OnlineRubrics 仍需单独实现。

### DPO 改造

1. 不重训练 EvoLM；下载已公开 `EvoLM-8B` 并冻结为 rubric producer。
2. 对共享 prompt `x` 用 EvoLM rubric-generation prompt 产生结构化 rubric `C_x`。
3. 使用同一共享 sampler 生成 `K>=4` 个 responses；用冻结 Qwen3-1.7B（或全部 auto-rubric 统一的 common judge）输出分项分和总分 `R_k`。
4. `chosen=argmax R_k`，`rejected=argmin R_k`，`m=R_chosen-R_rejected`；可同时保留 criterion score vector 供 `s` 校准和诊断。
5. 不使用 EvoLM 的“后期 checkpoint 天然优于早期 checkpoint”作 DPO chosen 标签；这只是它训 rubric generator 的自监督假设。我们的 DPO pair 应根据同一 rubric+judge 的实际 score 排序。

### 判定

- **三者中最适合单卡直接执行。** 16.4GB 权重可在 H100 80GB 上进行 BF16 推理，1.7B judge 也可顺序加载；数据生成无需闭源 API。
- **有官方代码不等于单卡适合复现完整共演化训练。** 对当前单张 H100 和 DPO 论文目标，实验名称仍应是 `DPO-margin + frozen EvoLM rubric generator`，而不是声称重现 EvoLM co-training。
- 执行时应固定官方 GitHub commit 和 HF model commit，并在重分发前单独核对代码/权重许可证。

## 共同公平适配协议

为了比较“不同 rubric producer + ours”而不是比较不同 API 或采样预算，三者必须共享：

- 同一 prompt train/calibration/eval split；
- 同一个冻结 response sampler、同一组预生成 responses；
- 尽可能同一个 pointwise rubric judge；若使用 native judge，必须加报 common-judge 结果；
- 同一 `chosen=top score / rejected=bottom score` 和 tie/min-gap filter；
- 同一 margin 归一化，建议在 train split 内把每个 producer 的 `m` 映射到可比的分位数尺度，避免某 producer 仅因分数范围更宽而占优；
- 每个 producer 在固定 calibration bank 上只计算一个 system-level `s(C)`；
- 同一 DPO-margin consumer（主实验建议先固定 MMPO）、同一 `rho`、seed 和训练 token budget。

## 排期优先级

1. **EvoLM-frozen：P0。** 有作者官方代码+权重，无需 API，可最快得到 rubric/criterion scores/pairs/m。
2. **RRD-EvoLM-reimplementation：P1。** 直接复用 EvoLM 官方仓中的 `rrd_wu` 管线，先用本地 proposer/judge 跑通，再视预算补 GPT-4o faithful subset；报告时明确其为后续复现。
3. **OnlineRubrics-offline-adapt：P1/P2。** 先实现 pairwise criteria elicitation 的离线版；不将单卡完整 online GRPO 复现放入主实验阻塞路径。
