# DPO-margin × Automatic Rubrics：实验基线与 Benchmark 审计

> **状态：历史调研材料（非执行规范）。** 本文件保留当时的候选方案与术语，可能与最终设计不同。正式实验只以根目录 [`PLAN.md`](../../PLAN.md)、[`CHECKLIST.md`](../../CHECKLIST.md) 和 [`final_experiment_matrix_audit.md`](../scout/final_experiment_matrix_audit.md) 为准；尤其不得从本文沿用旧称 “Scaled DPO”、旧 scorer/gate 或旧数据映射。

更新时间：2026-08-08

## 1. 结论先行

实验应拆成两个正交模块：

- **margin consumer（DPO-margin）**：给定 rubric 产生的标量分数差 (m_i)，决定怎样把“赢多少”放进 DPO loss。
- **rubric producer（automatic rubrics）**：产生固定、prompt-specific 或在线更新的 rubric；再由统一的冻结评分器产生 criterion-level 分数和 (m_i)。

推荐四个 DPO-margin 原型：

1. **MMPO**：把 (m_i) 转成 soft preference target；代表 soft-label 路线。
2. **ODPO**：把 (m_i) 作为 pair-specific logit offset；代表 margin-offset 路线。
3. **Scaled DPO**：用 (m_i) 调整样本权重；代表 reweighting 路线。
4. **ML-RDPO**：同时保持偏好排序并拟合 rating gap；代表 explicit gap-matching 路线。

主文建议报告 **MMPO、ODPO、Scaled DPO**，ML-RDPO 放附录。四者用于开发和完整消融；主文三者恰好覆盖把 (m_i) 放入 DPO 的三种最基本操作：soft target、logit offset、sample weight。ML-RDPO 额外引入 rating-gap regression term，使“只增加本文 robust layer”的归因稍复杂，因而更适合作为更强但次要的扩展基线。

自动 rubric 主线建议：

- **Auto-Rubric**：offline、全局/层次 rubric，最适合验证一个 rubric system 对应一个 (s(\mathcal C))。
- **OpenRubrics / Rubric-ARM**：公开数据和模型较完整；Rubric-ARM 原文直接做 DPO/IterDPO，是最成熟的 automatic-rubric × DPO 锚点。
- **Online Rubrics**：原生给出逐 criterion 的满足向量，和 (m_i)、(s_t) 最匹配；但原文策略优化是 GRPO，改成 iterative DPO 属于合理但必须明确标注的 adaptation。
- **RRD、EvoLM**：更适合作为扩展或附录。二者原生是 RL/GRPO 路线，直接纳入主要 DPO 表会同时改变 rubric 生成、训练算法、模型和数据，难以归因。

## 2. 必须使用的四行对照

仅比较“DPO-margin / +auto-rubric / +auto-rubric+ours”不足以隔离贡献。每个主要组合至少需要：

| 行 | 配置 | 回答的问题 |
|---|---|---|
| 1 | (B+\mathcal C_{source}) | 原始 DPO-margin 基线 |
| 2 | (B+A) | 自动生成/更新 rubric 本身是否有用 |
| 3 | (B+A+\text{robust}(s\equiv1)) | 一般 minimax/KL robustness 是否已经足够 |
| 4 | (B+A+\text{ours}(s(\mathcal C))) | rubric-conditioned uncertainty 是否带来额外收益 |

其中 (B) 是 DPO-margin consumer，(A) 是 automatic-rubric producer。第 3、4 行必须使用完全相同的固定常数 (\rho)，不能把 (\rho) 改成 rubric-specific budget。

若研究 native/static rubric，还应单独比较：

| 行 | 配置 |
|---|---|
| 1 | (B+\mathcal C_{native}) |
| 2 | (B+\mathcal C_{native}+\text{robust}(s\equiv1)) |
| 3 | (B+\mathcal C_{native}+\text{ours}(s(\mathcal C_{native}))) |

## 3. DPO-margin 候选

| 方法 | (m_i) 如何进入训练 | 推荐角色 | 原始/合适数据 |
|---|---|---|---|
| MMPO | (p_i=\sigma(\gamma m_i))，优化 soft binary target | 主基线；与“名义偏好分布”解释最自然 | UltraFeedback、SHP；可接任意 rubric score gap |
| ODPO | pair-specific logit offset | 主基线；最直接对应“减去偏置/间隔项” | 原文 TL;DR ratings；可迁移到 rubric gap |
| Scaled DPO | 用强度缩放样本 loss/gradient | 第四开发基线、附录 | HelpSteer2-Preference 最自然 |
| ML-RDPO | ranking loss + squared rating-gap matching | 主基线；机制与前三者不同 | UltraFeedback 或具有多级分数的数据 |

不建议把 RLCF 本身列为第五种 DPO-margin loss：RLCF 主要是 rubric-conditioned 的样本比较/构造过程，下游仍可使用标准 DPO。它更适合作为 rubric producer / preference-data construction baseline。

## 4. Automatic-rubric 方法审计

| 方法 | rubric 形态 | 原生策略优化 | 是否原生逐 criterion 打分 | 数据/benchmark | 建议定位 |
|---|---|---|---|---|---|
| Auto-Rubric | offline，全局层次 rubric | WildChat 上 DPO | 原生主要是 rubric-guided pairwise judge；需统一 pointwise adapter | rubric induction: HelpSteer3-Preference、UltraFeedback-Binarized；policy eval: Arena-Hard、AlpacaEval | 主实验：最干净的固定 (s(\mathcal C)) |
| OpenRubrics | offline、prompt-specific rules/principles | Qwen2.5-7B DPO | 原生重点是 pairwise Rubric-RM；绝对逐维分数需 adapter | OpenRubric-v2；IFEval、InfoBench、IFBench、Arena-Hard、AlpacaEval、WildBench、HealthBench | 公开且可复现的 offline 主基线 |
| Rubric-ARM | prompt-specific rubric generator + pairwise judge | DPO、IterDPO，也做 GRPO | 原生主要输出 pairwise decision/justification；需 adapter | OpenRubrics；IFEval、InfoBench、IFBench、Arena-Hard、AlpacaEval、WildBench、Creative Writing | 最成熟的 automatic-rubric × DPO 锚点 |
| Online Rubrics | 在线从 current/reference response 对中补充 criteria | 原文 GRPO | **是**，每个 response 的 criterion satisfaction vector | 自建 Generalist/Expert；AlpacaEval、Arena-Hard、GPQA、GSM8K | 在线主实验，但 DPO 版本属于 adaptation |
| RRD | recursive decomposition + filtering + correlation-aware weighting | RFT/Dr.GRPO | 可形成结构化 criteria，但原文重点 judge/reward，不是 DPO | JudgeBench、PPE；WildChat；BiGGen、HealthBench-Hard | 覆盖扩展压力测试/附录 |
| EvoLM | policy 与 discriminative rubrics 共同在线演化 | GRPO | rubric-item score 后聚合，概念上兼容 | Tulu3 mixture；通用 12 benchmarks；RewardBench2、JudgeBench | 高成本 stretch；不宜作为主 DPO baseline |

注意：名字“Auto-Rubric”可能与 2026 年的 “Autorubric: Unifying Rubric-based LLM Evaluation”混淆。本设计使用的是 *From Implicit Weights to Explicit Rubrics: A Training-Free Framework for Reward Modeling*。

## 5. 统一适配层：让所有 rubric producer 都能接四种 DPO-margin

为了避免把“rubric 质量”和“不同 judge 的能力”混在一起，主要 controlled experiment 应固定同一个 response sampler 和同一个 pointwise rubric grader：

1. 对同一批 prompt (x_i)，由固定 probe/current policy 采样 (M=8) 个回答 (y_{ij})。
2. automatic-rubric producer (A) 产生 (\mathcal C_i=\{c_{ik},w_{ik}\})。固定 rubric 方法则所有 prompt 共享 (\mathcal C)。
3. 同一个冻结 grader 对每个 ((x_i,y_{ij},c_{ik})) 输出 (v_{ijk})。
4. 聚合为
   \[
   R_{ij}=\frac{\sum_k w_{ik}v_{ijk}}{\sum_k |w_{ik}|}.
   \]
5. 取 (y_i^w=\arg\max_j R_{ij})、(y_i^l=\arg\min_j R_{ij})，并令
   \[
   m_i=R_i^w-R_i^l.
   \]
6. 将同一份 ((x_i,y_i^w,y_i^l,m_i)) 分别送给 MMPO、ODPO、Scaled DPO、ML-RDPO。
7. 用固定 calibration bank 计算 rubric-system scalar (s(\mathcal C))。offline 系统一个 (s)；prompt-specific 系统先逐 prompt 计算 operational information，再在 calibration bank 上聚合成一个 system-level (s)；online 系统每个 rubric 更新轮次一个 (s_t)，而不是每个样本一个 (s_i)。
8. 在相同固定 (\rho) 下比较 uniform robust (s=1) 与 ours (s(\mathcal C))。

该统一适配层对 OpenRubrics、Rubric-ARM、Auto-Rubric 尤其重要，因为它们原生更偏 pairwise decision。论文中应同时报告它们的 **native judge 结果** 和 **common-adapter 结果**；前者复现原方法，后者实现公平的 margin/s 比较。

## 6. 推荐实验轨道

### Track A：native rubric 的 DPO-margin 插件验证

目的：证明方法不是依赖某个 automatic-rubric generator。

- **UltraFeedback**：用其 4 个 aspect score（helpfulness、honesty、instruction-following、truthfulness）构造 (m_i)，跑四个 margin consumer；主表报告 MMPO、ODPO、Scaled DPO。
- **HelpSteer2 / HelpSteer2-Preference**：多维 ratings，重点跑 Scaled DPO、MMPO 和 ML-RDPO，验证从数值 rating 到 margin 的兼容性。
- **WildChecklists / RLCF-style data**：用 checklist/rubric satisfaction 构造多回答分数，验证更细粒度、prompt-specific rubrics。

每个数据集都比较 nominal、uniform robust、ours，并保持同一 (\rho)。

### Track B：offline automatic rubrics

主 controlled setting：

- policy/reference：Qwen2.5-7B-Instruct（与 Auto-Rubric、OpenRubrics、Rubric-ARM 的设置交集最大）。
- train prompts：4K–10K WildChat，或 OpenRubric-v2 的 general-domain 子集。
- generators：Auto-Rubric、OpenRubrics、Rubric-ARM。
- margin consumer：先固定 MMPO 做全 generator 比较；再只在最强 generator（预计 Rubric-ARM）上补 ODPO、Scaled DPO，避免全笛卡尔积。
- eval：IFEval、InfoBench、Arena-Hard style-controlled、AlpacaEval 2 length-controlled、WildBench。

最小主表：

| Rubric producer | MMPO | + uniform robust | + ours |
|---|---:|---:|---:|
| source/native rubric | ✓ | ✓ | ✓ |
| Auto-Rubric | ✓ | ✓ | ✓ |
| OpenRubrics | ✓ | ✓ | ✓ |
| Rubric-ARM | ✓ | ✓ | ✓ |

然后对 Rubric-ARM 额外跑 ODPO 和 Scaled DPO 三列，证明 ours 是 loss-agnostic plug-in。ML-RDPO 留作附录强基线。

### Track C：online / evolving rubrics

首选是把 Online Rubrics 的 criterion elicitation 接到 Rubric-ARM 风格的 IterDPO：

1. 第 (t) 轮由 current/reference policy 产生 response pairs。
2. Online Rubrics 更新 (\mathcal C_t)。
3. 统一 grader 得到 criterion scores、(m_i^{(t)})。
4. 计算固定 calibration bank 上的单一 (s_t=s(\mathcal C_t))。
5. 用 MMPO 或 ODPO 更新 policy；(\rho) 在所有轮次保持相同。

核心对照：固定初始 rubric、只在线更新 rubric、在线 rubric + uniform robust、在线 rubric + ours。记录 (s_t)、rubric 大小、训练轮次和下游性能随时间的变化。

EvoLM 只建议作为 stretch：若资源足够，可保留其 co-evolution rubric producer，但把策略更新替换为 MMPO；必须明确这不是原文复现。

## 7. 推荐 Benchmark 配置

| 目标 | 训练数据 | 主评测 | 原因 |
|---|---|---|---|
| static/native rubric | UltraFeedback、HelpSteer2-Preference、WildChecklists/RLCF data | Arena-Hard-SC、AlpacaEval2-LC、WildBench；加 rubric judge agreement | 原生有 aspect/rating/checklist 信号，可直接构造 (m) 和 (s) |
| offline auto-rubric | WildChat 4K–10K 或 OpenRubric-v2 general-domain | IFEval、InfoBench、Arena-Hard-SC、AlpacaEval2-LC、WildBench | 与三种 offline/ARM 方法重叠最多，能统一 Qwen2.5-7B |
| online auto-rubric | 同一 WildChat 子集；另加 Expert subset stress test | 通用集 + GPQA-Diamond/GSM8K 或 HealthBench-Hard | 检查 rubric 更新是否补充任务覆盖，而不是只扩大 rubric 文本 |
| rubric/judge quality | RewardBench2、JudgeBench；criterion omission synthetic tests | pair accuracy、calibration、preference flip、(s) 与受控遗漏程度的单调性 | 验证 (s) 的含义，而非只看最终 policy 胜率 |

## 8. 训练与公平性约束

- 所有对照共享 prompts、候选 responses、grader、reference policy、训练 token budget 和随机种子。
- (m_i) 做数据集内稳健归一化，并为四种 consumer 固定同一归一化结果；不要为每种 loss 单独重新定义“赢多少”。
- (s(\mathcal C)) 只随 rubric system / dataset（或 online round）变化，不随单个训练样本变化。
- (\rho) 是预设常数；先用一个独立 pilot/validation 选择，之后在所有 rubric 与 baseline 上固定。
- 对 online rubric，固定 calibration prompt-response bank，防止 (s_t) 的变化其实来自 policy response 分布变化。
- 报告 API/judge 成本、rubric 数量、平均 criterion 数、训练 wall-clock；RRD/EvoLM 的成本可能决定它们只能进入附录。

## 9. 最终建议的主文规模

为了避免不可解释的 (4\times6) 笛卡尔积：

1. **四个 DPO-margin 开发，三个主文报告**：MMPO、ODPO、Scaled DPO；ML-RDPO 附录。
2. **三个 automatic-rubric 主对象**：Auto-Rubric、Rubric-ARM、Online Rubrics；OpenRubrics 作为 Rubric-ARM 的公开前身/复现锚点，至少保留一张结果表。
3. **主 controlled generator 比较固定 MMPO**；只在 Rubric-ARM 上横向替换 ODPO、Scaled DPO。
4. **核心数据**：UltraFeedback（native rubric）、OpenRubric-v2/WildChat（offline auto rubric）、WildChat iterative subset（online）。HelpSteer2 和 WildChecklists 用于补充泛化。
5. 每张核心表都包含 uniform robust (s=1)，否则无法支持“贡献来自衡量 unmeasured rubric confounding”的主张。

## 10. 主要来源

- Auto-Rubric: https://arxiv.org/html/2510.17314
- Auto-Rubric dataset: https://huggingface.co/datasets/agentscope-ai/Auto-Rubric
- RRD: https://arxiv.org/html/2602.05125
- OpenRubrics: https://arxiv.org/html/2510.07743
- OpenRubric-v2: https://huggingface.co/datasets/OpenRubrics/OpenRubric-v2
- Rubric-ARM: https://arxiv.org/html/2602.01511
- Rubric-ARM rubric generator: https://huggingface.co/OpenRubrics/RubricARM-8B-Rubric
- Rubric-ARM judge: https://huggingface.co/OpenRubrics/RubricARM-8B-Judge
- Online Rubrics: https://arxiv.org/html/2510.07284
- EvoLM: https://arxiv.org/abs/2605.03871
- Public Chasing-the-Tail rubrics fallback: https://huggingface.co/datasets/JunkaiZ/Rubrics
