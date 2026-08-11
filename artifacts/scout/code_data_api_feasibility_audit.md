# Rubric producer × DPO-margin：代码、数据与 API 可行性总审计

> **状态：历史调研材料（非执行规范）。** 本文件保留当时的候选方案与术语，可能与最终设计不同。正式实验只以根目录 [`PLAN.md`](../../PLAN.md)、[`CHECKLIST.md`](../../CHECKLIST.md) 和 [`final_experiment_matrix_audit.md`](../scout/final_experiment_matrix_audit.md) 为准；尤其不得从本文沿用旧称 “Scaled DPO”、旧 scorer/gate 或旧数据映射。

> 审计日期：2026-08-10。方法事实只采用论文、作者仓库、作者模型/数据卡和官方框架文档。这里的“可运行”表示存在明确资源或实现路径，不表示已经在当前 H100 上完成 smoke test。

## 1. 结论

实验可以做，而且主 controlled track **不必调用商业 API**。但六个 rubric 方法的开放程度不同，论文中必须区分三类 provenance：

1. **官方代码/权重直接使用**：OpenRubrics、Rubric-ARM、EvoLM；
2. **官方数据或论文 rubric 直接使用，但生成算法需自行实现**：Auto-Rubric；
3. **论文方法的本地适配/后续复现**：OnlineRubrics、RRD。

六者都可以输出 rubric；但只有 OnlineRubrics、RRD、EvoLM 原生定义了逐回答的 rubric aggregate score。Auto-Rubric、OpenRubrics、Rubric-ARM 主要给 pairwise verdict。因此主公平实验统一采用一个冻结的本地 pointwise criterion grader，把每个 producer 的 rubric 转成 criterion-score vector、aggregate score、winner/loser 和连续 margin。这样比较对象是 rubric，而不是六套不同 judge。

## 2. DPO / margin consumer 代码状态

| 方法 | 官方资源 | 状态 | 本文实现决定 |
|---|---|---|---|
| Vanilla DPO | [论文](https://arxiv.org/abs/2305.18290)、[作者代码](https://github.com/eric-mitchell/direct-preference-optimization) | 有完整 Apache-2.0 参考实现，但依赖和 checkpoint 格式较旧 | 用当前 [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer) 作统一基础设施；用作者实现做数值对齐 |
| MMPO | [论文](https://aclanthology.org/2024.findings-emnlp.792/)、[作者代码](https://github.com/kykim0/margin-matching-pref-opt) | 有 Apache-2.0 代码、UltraFeedback/SHP recipe 和连续 score-gap 接口；依赖较旧 | 主 margin consumer；在统一 trainer 中复现 soft target，并逐样本对齐作者 loss |
| ODPO | [论文](https://aclanthology.org/2024.findings-acl.592/)、[作者代码](https://github.com/rycolab/odpo) | 有 Apache-2.0 代码；原 recipe 是 IMDB、toxicity、TL;DR，不支持当前聊天数据开箱即用 | 在统一 trainer 重实现 offset loss；`alpha=0` 必须退化为 DPO |
| Scaled DPO | [HelpSteer2-Preference 论文](https://arxiv.org/abs/2410.01257)、[官方数据](https://huggingface.co/datasets/nvidia/HelpSteer2) | 论文公式和数据公开；未发现作者发布的 Scaled-DPO 专用训练入口 | 按论文公式做 sample-weight loss；明确写成 formula reproduction，不声称使用官方训练代码 |

四套旧框架不分别维护。统一 schema 为 `prompt/chosen/rejected/margin_raw/margin_normalized/rubric_system_id`，所有 loss 共享 tokenizer、reference log-prob、optimizer、batch、seed 和 checkpoint rule。

## 3. 六个 automatic-rubric producer

| Producer | 官方开放资产 | 原生输出 | DPO/pair 可用性 | 商业 API | 本文名称与地位 |
|---|---|---|---|---|---|
| Auto-Rubric | [论文](https://arxiv.org/abs/2510.17314)、[38,459 条官方数据](https://huggingface.co/datasets/agentscope-ai/Auto-Rubric)；未发现官方生成代码或模型 checkpoint | query-specific rubrics、最终 hierarchical Theme–Tips rubric、binary pair verdict | 官方数据已含两个回答和 chosen/rejected；新 prompt 生成需按附录实现 | 非必需；论文用开源 Qwen backbone 也可做 | `Auto-Rubric (released rubric/data)`；主比较可用发布 rubric，生成算法复现为次要任务 |
| OpenRubrics | [论文](https://arxiv.org/abs/2510.07743)、[官方代码](https://github.com/wanghaoyu0408/OpenRubrics)、[OpenRubric-v2](https://huggingface.co/datasets/OpenRubrics/OpenRubric-v2)、4B/8B generator 和 judge | prompt-specific Hard Rules/Principles、pairwise winner | 74.2k 发布数据可直接转 binary DPO；本地 checkpoint 可给新 prompt 造 rubric/pair | 使用发布 checkpoint 不需要；忠实重造原始语料需要 GPT/Gemini 等 API | `OpenRubrics/RubricRM-v2`；P0 主比较 |
| Rubric-ARM | [论文](https://arxiv.org/abs/2602.01511)、[官方代码](https://github.com/wanghaoyu0408/OpenRubrics/tree/main/rubric-arm)、[8B generator](https://huggingface.co/OpenRubrics/RubricARM-8B-Rubric)、[8B judge](https://huggingface.co/OpenRubrics/RubricARM-8B-Judge) | prompt-specific rubric、双顺序 pairwise winner | 原方法已有 sample → dual-order judge → filter → DPO/IterDPO 路径 | 使用发布权重不需要 | `Rubric-ARM released checkpoints`；P0、预注册 consumer plug-in 锚点 |
| OnlineRubrics | [论文](https://arxiv.org/abs/2510.07284)、[机构页](https://scale.com/research/onlinerubrics)；未发现官方方法代码或 Generalist/Expert 数据下载 | current/control response comparison 诱导 criteria；逐项 0/1 和 weighted reward | 能直接由多个回答的 reward 构造 pair/m；需实现 extractor、dedup、grader orchestration | 忠实版需要 o3-mini/GPT-4.1-mini；本地版不需要 | `OnlineRubrics-local/offline-adapt`；P1，另做 2–3 轮 online extension |
| RRD | [论文](https://arxiv.org/abs/2602.05125)；无 RRD 原作者代码/checkpoint；[EvoLM 官方仓](https://github.com/stellalisy/EvoLM)含 RRD-WU/LLM/uniform 后续复现 | recursive decomposition/filter、逐项 0/1、correlation-aware aggregate reward | aggregate score 可直接构造 pair/m | faithful proposer 用 GPT-4o；本地 proposer/judge 不需要 | `RRD-local (EvoLM authors' reimplementation)`；P1，不能写成 RRD official code |
| EvoLM | [论文](https://arxiv.org/abs/2605.03871)、[作者代码](https://github.com/stellalisy/EvoLM)、[EvoLM-8B](https://huggingface.co/stellalisy/EvoLM-8B/tree/main) | JSON criteria/weights/scoring levels、逐项和总分 | 冻结 producer + local judge 后可直接构造 pair/m | 主方法不需要 | `frozen EvoLM rubric generator + DPO-margin`；P0。完整共演化不是单 H100 主任务 |

“有代码”不等于适合单卡完整重训：Rubric-ARM 的 alternating RL、OnlineRubrics 的 online GRPO、EvoLM 的完整 co-evolution 都远重于本文需要。本文只把 producer/judge 作为冻结的数据构造插件；这是与 DPO-margin 公平组合的 adaptation，不冒充原论文完整训练复现。

## 4. 数据集可行性

| 数据 | rubric/score 信号 | preference 信号 | 角色 | 判断 |
|---|---|---|---|---|
| [UltraFeedback raw](https://huggingface.co/datasets/openbmb/UltraFeedback) + [cleaned binarized](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) | 四个 aspect scores | chosen/rejected | native track | 可用；由 aspect score 差重建 `m`，不用有问题的复制 `overall_score` 字段 |
| [WildChecklists](https://huggingface.co/datasets/viswavi/wildchecklists) / [RLCF code](https://github.com/viswavi/RLCF) | 51.1k rows；prompt-specific requirements、importance、chosen/rejected score | 已有 chosen/rejected | native track | 最完整的 rubric-DPO 数据/代码锚点，可直接用发布分数，不必重新跑 72B judge |
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | 21,362 responses；helpfulness/correctness/coherence/complexity/verbosity 五项 0–4 人评分 | 两回答同 prompt；另有 -3…3 preference strength 和 justification | 第三个 native track | 强烈加入；无需 API、无需新增人工标注，还能用 human strength 做独立验证 |
| [Auto-Rubric data](https://huggingface.co/datasets/agentscope-ai/Auto-Rubric) | query-specific rubric text | 两回答和 binary label | released-artifact reproduction | 可直接做 binary DPO；连续 `m` 需统一 scorer |
| [OpenRubric-v2](https://huggingface.co/datasets/OpenRubrics/OpenRubric-v2) | rubric + judge trajectory | response A/B + winner | released-artifact reproduction | 可直接 binary DPO；连续 `m` 需统一 scorer |
| [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | 无 | 无固定 pair | six-producer common controlled track | 最合适：Auto-Rubric/RRD/RLCF 的策略实验都与 WildChat 有交集；本地采样共同 responses 后公平造 pair |
| [JudgmentBench](https://huggingface.co/datasets/judgmentbench/JudgmentBench) | 同一输出有 task-specific weighted rubric 和 1,539 expert rubric annotations | 1,530 expert pairwise judgments | problem/measurement diagnostic，不训练 | 极有价值：同一输出池同时有 rubric score 与独立 preference，可直接检验“rubric margin 不等于完整偏好” |

ResearchRubrics、Prometheus Feedback Collection 可作为领域/评分扩展，但不是比上述三套 native 数据更干净的核心 DPO 训练集。JudgmentBench 规模也不足以承担主训练，只做独立诊断。

## 5. 不调用 API 的统一造数协议

主 controlled track 固定 WildChat prompt pool，并一次性缓存同一批本地 responses：

1. 对每个 prompt，用冻结 `Qwen/Qwen2.5-7B-Instruct` 采样 `M_train=8` 个回答；六个 producers 共用。该数量与 RRD 的 rubric-proposal 设置对齐，也支持 OnlineRubrics 的 pairwise elicitation。
2. 每个 producer 生成其完整 rubric。Auto-Rubric 使用发布的完整 rubric；OpenRubrics、Rubric-ARM、EvoLM 用发布 checkpoint；RRD 用 EvoLM 仓 reimplementation；OnlineRubrics 以 RLCF 发布的完整 universal requirements 为固定初始 rubric，再按论文 prompt 做本地 criteria-elicitation adaptation。
3. 用一个冻结的本地 pointwise criterion grader 对 `(prompt,response,criterion,anchors)` 打分。首选 [Prometheus-7B-v2](https://github.com/prometheus-eval/prometheus-eval)，它官方支持本地 vLLM 和 1–5 absolute grading。
4. 按 producer 的公开 importance/aggregation rule得到

   \[
   R_{gij}=\frac{\sum_k w_{gik}z_{gijk}}{\sum_k w_{gik}}.
   \]

5. 取 `chosen=argmax_j R_gij`、`rejected=argmin_j R_gij`，定义 `m_raw=R_chosen-R_rejected`；预注册 tie/min-gap filter。
6. 每个 producer 的 margin normalization 只在其 train split 拟合，并映射到同一 quantile scale；保存 raw 和 normalized 两列。
7. calibration bank 与 train/val 完全不重叠；每个 producer 最终只得到一个 system-level `s_g`。
8. 输出 TRL preference schema；[TRL 官方格式](https://huggingface.co/docs/trl/dataset_formats)只要求 prompt/chosen/rejected，本文额外保留 margin 与 system metadata。

这个协议不训练新 reward network，也不需要商业 API。方法 native judge 的 binary/pairwise 结果只放 reproduction/敏感性表；若要把 pairwise verdict 转成连续 margin，使用双顺序重复判断 + prompt-local Bradley–Terry utility，不能把单次 0/1 winner 假装成 calibrated gap。

## 6. 最终实验矩阵

### A. Native rubric 主表

列为 UltraFeedback、WildChecklists、HelpSteer2。行包括 Base、DPO，以及 MMPO/ODPO/Scaled DPO 各自的 nominal、`uniform robust (s=1)`、ours。uniform 行不可删除，否则无法判断收益来自一般 robust learning 还是 rubric-conditioned `s`。

### B. Six-producer controlled table

只用共同 WildChat data，先固定 MMPO：

| Producer | producer-specific DPO | MMPO nominal | uniform robust | ours |
|---|---:|---:|---:|---:|
| Auto-Rubric released rubric | yes | yes | yes | yes |
| OpenRubrics/RubricRM-v2 | yes | yes | yes | yes |
| Rubric-ARM | yes | yes | yes | yes |
| OnlineRubrics-local | yes | yes | yes | yes |
| RRD-local (EvoLM reimplementation) | yes | yes | yes | yes |
| frozen EvoLM | yes | yes | yes | yes |

各 producer 可能从同一 response bank 选出不同 winner/loser，因此 DPO 也必须是 producer-specific：`DPO_g` 使用该 producer 的 pair 但忽略 `m`。不能用一个含义不清的“共享 DPO”代替六个内部二元下界。

不要做 `6 producers × 3 consumers × 3 datasets` 的全笛卡尔积。DPO-loss 泛化已经由 native track 证明；automatic track 的首要变量应是 rubric producer。预注册 Rubric-ARM 作为 consumer plug-in 锚点，再补 ODPO/Scaled nominal、uniform、ours。

### C. 多 system 联合识别

把六个 natural rubric systems 的等量数据放在同一训练中，共享一个全局固定 `rho`。比较 nominal、uniform、correct `s_g`、system-level shuffled `s_g`、inverse `s_g`。这是区分 rubric-conditioned weighting 与单纯调整 effective radius 的核心实验。

### D. 方法原生/动态扩展

- 发布数据复现：Auto-Rubric data binary DPO、OpenRubric-v2 binary DPO、Rubric-ARM dual-order DPO；结果不跨数据集横比。
- OnlineRubrics：offline producer sweep 之外，再做 2–3 轮 iterative DPO；每轮一个 `s_t`、全程同一 `rho`。
- EvoLM：主表使用 frozen producer；完整 co-evolution 不进入单 H100 阻塞路径。
- RRD：主表为 local/EvoLM-reimplementation；预算允许时只在 200 prompts 上补 GPT-4o faithful sensitivity。

## 7. 执行优先级

1. 统一 DPO/MMPO/ODPO/Scaled loss 与数值单测。
2. WildChecklists 和 HelpSteer2 直接数据 smoke；UltraFeedback 完成 raw/binarized 对齐。
3. Rubric-ARM、OpenRubrics、EvoLM 三个官方 checkpoint 的 common-WildChat pilot。
4. Auto-Rubric released rubric 接入。
5. RRD-EvoLM reimplementation。
6. OnlineRubrics-local；最后才做 genuine online rounds。

若 P0/P1 适配器未通过 parser、pair consistency 和 score-validity gate，不应先启动大规模 DPO 训练。
