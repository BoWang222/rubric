# 通用 Benchmark 之前：DPO/Rubric 实验执行计划（历史草案）

> **状态：历史调研材料（非执行规范）。** 本文件保留当时的候选方案与术语，可能与最终设计不同。正式实验只以根目录 [`PLAN.md`](../../PLAN.md)、[`CHECKLIST.md`](../../CHECKLIST.md) 和 [`final_experiment_matrix_audit.md`](../scout/final_experiment_matrix_audit.md) 为准；尤其不得从本文沿用旧称 “Scaled DPO”、旧 scorer/gate 或旧数据映射。

> 该文件保留作调研记录。当前可执行计划以仓库根目录 `PLAN.md` 和 `CHECKLIST.md` 为准；其中已删除 1D/2D/3D rubric omission，并增加自然多-rubric system 联合训练。

更新时间：2026-08-10

## 1. 训练阶段结论：主实验不重新 SFT

主 controlled experiment 使用同一个已经 instruction-tuned / SFT 的 checkpoint 作为：

1. 未训练的 `Base/SFT` 对照；
2. 所有 DPO-margin 方法的初始化；
3. 冻结 reference policy `pi_ref`。

推荐主模型：`Qwen/Qwen2.5-7B-Instruct`。它与 RLCF、Auto-Rubric、OpenRubrics、Rubric-ARM、Online Rubrics 的实验交集最大。

不重新训练 SFT 的原因：本文是增量式 DPO/rubric robustness 工作；重新 SFT 会引入新的 SFT 数据、训练预算与 checkpoint 差异，使方法收益无法归因。

论文复现与最终 controlled comparison 要区分：

- MMPO 原论文自己训练了 SFT checkpoint，再做 alignment；其官方代码提供 SFT/DPO recipe。
- ODPO 原文从任务对应的 SFT checkpoint 开始训练 ODPO。
- HelpSteer2-Preference 的 DPO 实验直接从 Llama-3.1-70B-Instruct 开始，没有另做 policy SFT。
- RLCF、Auto-Rubric、OpenRubrics、Rubric-ARM 从 Qwen2.5-7B-Instruct 开始做 DPO/IterDPO。
- Rubric-ARM 的 DPO objective 使用 0.2 的 SFT auxiliary mixing weight；这不是独立 SFT stage。复现该轨道时，所有比较行必须共同使用相同的 0.2 mixing，或在 controlled ablation 中共同关闭。

因此：作者配置只用于 implementation sanity check；论文主表统一使用同一个起点，只训练 DPO 阶段。

## 2. 需要收集的数据

### 2.1 核心训练数据

| 数据 | 用途 | 必须保留的字段 | 来源 |
|---|---|---|---|
| 原始 UltraFeedback | 多回答、4D rubric、计算 margin 与 s | instruction、每个 response、overall/aspect scores、explanations | `openbmb/UltraFeedback` |
| UltraFeedback Binarized | 共同 DPO pair benchmark | prompt、chosen、rejected、chosen/rejected score | `HuggingFaceH4/ultrafeedback_binarized` 或 `trl-lib/ultrafeedback_binarized` |
| HelpSteer2 / Preference | 人类 5D rating、1/2/3 preference strength、nested-rubric omission | prompt、response_1/2、5D ratings、preference strength、split | `nvidia/HelpSteer2` 的 preference 配置 |
| WildChecklists | 真实 prompt-specific checklist-DPO | prompt、checklist、importance、chosen/rejected、criterion/total scores | `viswavi/wildchecklists` |
| OpenRubric-v2 | offline automatic rubric | instruction、rubric、response pairs/listwise candidates、source、judge label | `OpenRubrics/OpenRubric-v2` |
| Auto-Rubric | offline global/hierarchical rubric 及已生成数据 | prompt、rubric/system id、response/pair fields | `agentscope-ai/Auto-Rubric` |
| WildChat prompt subset | offline/online auto-rubric controlled prompts | prompt id、prompt text、source/safety metadata | 优先复用 RLCF 发布的 WildChat prompt ids，避免 split 不一致 |

### 2.2 Rubric producer 模型与数据

| Producer | 获取对象 | 说明 |
|---|---|---|
| Auto-Rubric | 官方数据和论文 prompt/算法 | 论文未发现独立官方训练代码仓；按论文实现 offline induction/compression，优先复用发布 rubric |
| OpenRubrics | OpenRubric-v2、公开 rubric generator/RM checkpoint（若对应版本可用） | 论文未给出独立 GitHub 训练仓；数据可直接使用 |
| Rubric-ARM | `OpenRubrics/RubricARM-8B-Rubric`、`OpenRubrics/RubricARM-8B-Judge` | 原文明确 DPO/IterDPO 基于 LLaMA-Factory，GRPO 基于 ms-swift |
| Online Rubrics | 论文附录完整 extractor/grader prompts | 原文未发现独立官方代码；使用论文 prompt 做方法复现，DPO 版本必须标注 adaptation |

### 2.3 数据版本记录

每个数据集下载后冻结：dataset repository、revision/commit、config、split、样本数、license、下载日期、SHA256/Arrow fingerprint。禁止实验中途让 `load_dataset` 自动跟随最新 revision。

## 3. 需要收集的 baseline 代码

### 3.1 最终统一训练栈

使用 `huggingface/trl` 的 `DPOTrainer` 作为最终 controlled implementation，固定版本/commit，并在其上实现：

- vanilla DPO；
- MMPO soft target；
- ODPO logit offset；
- Scaled DPO sample weighting；
- ML-RDPO（附录）；
- uniform-KL robust wrapper，`s=1`；
- ours，`s=s(C_g)`；
- optional SFT auxiliary loss mixing（只在复现 Rubric-ARM 配置时使用，同一轨道全方法相同）。

原因：最终论文比较必须共享模型加载、tokenization、reference log-prob、batching、optimizer、数据顺序和 logging。不能把不同作者仓库直接跑出的结果当作严格横向比较。

### 3.2 作者代码只用于复现检查

| 方法 | 官方/参考代码 | 用途 |
|---|---|---|
| DPO | `huggingface/trl`；原始 `eric-mitchell/direct-preference-optimization` | 验证标准 DPO logit/loss |
| MMPO | `kykim0/margin-matching-pref-opt` | 验证 margin-to-soft-target、gamma、UltraFeedback recipe |
| ODPO | `rycolab/odpo` | 验证 offset 位置、alpha、对称正反标签 loss |
| Margin/Scaled DPO | `NVIDIA/NeMo-Aligner`、HelpSteer2-Preference 论文公式 | 验证离散 margin 与 sample weighting |
| RLCF | `viswavi/RLCF` | checklist generation、candidate scoring、pair construction、OpenRLHF train/eval scripts |
| Rubric-ARM | `hiyouga/LlamaFactory` + Rubric-ARM checkpoints | 复现 DPO/IterDPO 数据接口和 SFT mixing |
| ML-RDPO | 论文公式；如无可用官方仓则在统一 trainer 实现 | 附录强 baseline |

作者代码的目标是通过小规模结果/逐样本 loss 对齐，不是作为最终四套训练框架。

## 4. 统一数据格式

所有 DPO 训练数据转成同一 schema：

```text
pair_id
dataset_id
rubric_system_id
prompt
chosen
rejected
score_chosen
score_rejected
margin_raw
margin_normalized
criterion_scores_chosen
criterion_scores_rejected
rubric_text_or_id
split
```

calibration sidecar 单独保存：

```text
rubric_system_id
prompt_id
response_id
criterion_id
criterion_weight
criterion_score
probe_policy_id
grader_id
sampling_seed
```

DPO training、s calibration、validation 和最终 specialized evaluation 的 prompt 不得重叠。

## 5. Phase 0：基础设施与 loss 正确性

### 5.1 固定实验合同

- 主 policy/reference：Qwen2.5-7B-Instruct；exact revision 固定。
- 主训练：full-parameter DPO；LoRA 只用于 smoke/pilot，不能与 full-tuning baseline 混表。
- max sequence length：2048。
- 相同 chat template、tokenizer、padding/truncation、optimizer 和 effective batch。
- 每个数据集内部所有方法共享相同 pair 顺序与 seeds。
- reference model 冻结；可预计算并缓存 chosen/rejected reference log-prob。
- rho 在独立 validation/pilot 上选择一次，之后对所有 rubrics、baseline 和 seeds 固定。

### 5.2 必做单元测试

1. ODPO 在 `alpha=0` 时逐样本等于 DPO。
2. Scaled DPO 在所有 sample weight 为 1 时等于 DPO。
3. uniform robust 与 ours 在 `s=1` 时逐样本/逐 batch 相等。
4. robust radius 为 0 时退化为对应 nominal margin baseline。
5. 正反 preference branch 对称，交换 chosen/rejected 后 loss 行为正确。
6. 同一 batch 在作者实现与统一实现中的 MMPO/ODPO loss 数值对齐。
7. s 只由 `rubric_system_id` 索引，同一 rubric 内不会随 pair 改变。

### 5.3 Smoke test

先用 UltraFeedback 1k--2k pairs、单 seed、短训练跑：Base、DPO、MMPO、ODPO、Scaled DPO、MMPO+uniform、MMPO+ours。检查 loss、reward margin、KL、response length、NaN、checkpoint/eval pipeline。

## 6. Phase 1：UltraFeedback——共同 DPO-margin benchmark

### 6.1 数据准备

1. 下载 raw UltraFeedback 与 binarized preference data。
2. 用 pair id / prompt / response text 对齐 raw aspect scores 与 chosen/rejected。
3. 去除无法对齐样本、空回答、重复回答、长度超限样本。
4. 主表使用 `m_i>0`；ties 进入 MMPO soft-label 附录。
5. 定义完整 4D score aggregation，并在 train split 内固定 margin normalization；validation/test 只应用该变换。

### 6.2 计算 rubric-level s

- 快速原型可使用原始每题 4 个回答的 4D score matrix。
- 正式实验在 held-out calibration prompts 上由冻结 Qwen2.5-7B-Instruct 采样 M=8 个回答，用冻结 grader 按完整评分说明得到 4D criterion matrix。
- 构造 1D/2D/3D/4D nested rubric systems；每个 system 只得到一个 `s_g`。
- 检查 s 与 rubric dimension、preference flip、相对 4D pseudo-oracle discrepancy 的单调/相关关系。

### 6.3 训练行

完整 margin comparison：

1. Base/SFT checkpoint（不训练）；
2. vanilla DPO；
3. MMPO；
4. ODPO；
5. Scaled DPO；
6. ML-RDPO（附录）；
7. MMPO + uniform robust (`s=1`)；
8. MMPO + ours (`s=s_g`)。

插件泛化只在完整 4D rubric 或最强设置补：ODPO + uniform/ours、Scaled DPO + uniform/ours。不需要在所有 nested rubrics 上做全笛卡尔积。

### 6.4 Gate

只有当 vanilla DPO 和 MMPO 的趋势与官方 recipe 接近、s 在 nested rubrics 上有合理区分、uniform 与 ours 数值实现通过测试，才进入 RLCF 和 auto-rubric。

## 7. Phase 2：HelpSteer2——受控 rubric omission

1. 使用公开 7,118 个 preference pairs（train/validation 按官方 split）。
2. 保留五维 ratings 和 1/2/3 preference strength，不新增人工标注。
3. 构造 5D、4D、3D、2D、1D rubric systems；固定预注册的维度顺序，并增加若干随机顺序作为 robustness check。
4. 完整 5D 只作为 pseudo-oracle，低维训练不访问被删除维度。
5. 核心训练只跑 MMPO、MMPO+uniform、MMPO+ours；Regular/Margin/Scaled DPO 用来复现数据集原生 baseline。
6. 报告 s 与 omission level、preference flip rate、pseudo-oracle discrepancy、ours-over-uniform gain 的跨-rubric 关系。

该阶段是证明 s 真正在衡量 rubric completeness 的主识别实验，而不是最终 policy benchmark。

## 8. Phase 3：RLCF/WildChecklists——真实 pipeline transfer

### 8.1 先使用预计算数据

第一轮不重新生成全部 WildChat checklist 和评分，直接使用 RLCF 发布的：

- checklists；
- Qwen2.5-7B-Instruct response pairs；
- Qwen2.5-72B-Instruct rubric scores；
- final offline preference data。

先复现 `RLCF + standard DPO`。RLCF 官方仓库提供 checklist generation、rubric judge、pair construction、OpenRLHF training 和评测代码。

### 8.2 s calibration sidecar

只对 held-out 500--1,000 个 calibration prompts 重新采样 M=8 responses，而不是重做全量训练数据。沿用 RLCF checklist generator、criterion judge 和 aggregation，计算一个系统级 `s_RLCF`。训练集中所有 pair 共享它。

### 8.3 训练行

1. Base Qwen2.5-7B-Instruct；
2. RLCF + DPO；
3. RLCF + MMPO；
4. RLCF + ODPO；
5. RLCF + Scaled DPO；
6. RLCF + MMPO + uniform robust；
7. RLCF + MMPO + ours。

主配置跟随 RLCF 的数据与训练预算；同一轨道内所有方法共享它。不要将 RLCF 的上游数据生成差异混入 loss comparison。

## 9. Phase 4：offline automatic rubrics

### 9.1 Controlled data generation

固定同一批 4k--10k WildChat prompts、同一冻结/current policy、同一 M=8 responses。分别运行：

- source/static rubric；
- Auto-Rubric；
- OpenRubrics；
- Rubric-ARM。

所有 producer 输出的 criteria 都通过同一个冻结 pointwise grader，得到 criterion-level scores、总分、winner/loser 和 margin。原方法的 native pairwise judge 结果可另表报告，但不能和 common-adapter 结果混为一列。

### 9.2 system-level s

- Auto-Rubric 的固定层次 rubric：一个 s。
- prompt-specific OpenRubrics/Rubric-ARM：将 generator + rubric + judge + aggregation 视为一个 measurement system，在同一 calibration bank 上聚合成一个 system-level s。
- 不生成 sample-dependent s。

### 9.3 训练矩阵

先固定 MMPO：

| Producer | MMPO | + uniform | + ours |
|---|---:|---:|---:|
| source/static | yes | yes | yes |
| Auto-Rubric | yes | yes | yes |
| OpenRubrics | yes | yes | yes |
| Rubric-ARM | yes | yes | yes |

再只在最强 producer（预期 Rubric-ARM）上补 ODPO 和 Scaled DPO 的 nominal/uniform/ours，证明 robust module 与 margin consumer 解耦。

Rubric-ARM controlled setup建议沿用其 DPO anchor：1 epoch、batch 64、LR 8e-7、beta 0.1、max length 2048；若保留 0.2 SFT mixing，则所有行共同保留。

## 10. Phase 5：online rubrics / IterDPO

offline pipeline 稳定后再做：

1. t=0 从同一 Qwen2.5-7B-Instruct 开始；
2. current/reference policy 在固定训练 prompt pool 上采样；
3. Online Rubrics 更新 `C_t`；
4. common grader 产生 scores 与 `m_i^(t)`；
5. 在从未变化的 calibration bank 上计算一个 `s_t=s(C_t)`；
6. 用 MMPO IterDPO 更新一个 round；
7. 重复 2--3 rounds。

四个控制：固定初始 rubric、只更新 rubric、在线 rubric + uniform robust、在线 rubric + ours。rho 所有轮次相同。Online Rubrics 原文是 GRPO，因此这里必须明确称为 DPO adaptation，而非原文复现。

## 11. 在通用 Benchmark 之前要完成的评测

每个训练阶段先完成与论文问题直接相关的 specialized evaluation：

- policy alignment：Arena-Hard style-controlled、AlpacaEval2 length-controlled、WildBench；
- instruction following：IFEval、InfoBench，RLCF 轨道增加 FollowBench；
- rubric/judge：RewardBench2、JudgeBench 或对应原论文 preference benchmark；
- method validity：s stability、nested-rubric monotonicity、preference flip、pseudo-oracle discrepancy；
- training diagnostics：train/val loss、DPO reward margin、policy-reference KL、response length、gradient norm、dual/inner-max status；
- efficiency：GPU hours、judge calls、rubric generation cost、training tokens。

只有 specialized metrics 与 diagnostics 合格的 checkpoint 才进入 MMLU/GSM8K/BBH/HumanEval/GPQA-Diamond 通用评测，避免在明显失败的 checkpoint 上浪费评测成本。

## 12. 调参与 seeds

- 第一阶段全部配置单 seed 筛选。
- hyperparameters 只在 validation 上选择；测试 benchmark 不用于挑 checkpoint。
- 每个 baseline 给予相同数量的调参 trial。
- 主表对 Base 以外的关键行至少 3 seeds：DPO、MMPO、uniform robust、ours；资源允许时对 ODPO/Scaled DPO 也做 3 seeds。
- rho 只选择一次并跨 rubrics 固定。
- alpha、gamma 与 weight normalization 可按 loss 在 validation 校准，但调参预算相同。
- 报告 mean、95% CI、response length；judge benchmark 还应报告 order-flip robustness。

## 13. 推荐执行顺序与停止门槛

1. 冻结代码、模型、数据 revision 和 schema。
2. 完成 DPO/MMPO/ODPO/Scaled/robust 单元测试。
3. UltraFeedback 1k smoke。
4. UltraFeedback full + nested-rubric s validation。
5. HelpSteer2 omission experiment。
6. RLCF precomputed-data reproduction + s sidecar。
7. offline automatic-rubric controlled table。
8. Rubric-ARM 上做 consumer plug-in 泛化。
9. Online Rubrics/IterDPO。
10. specialized benchmark、error analysis、3-seed promotion。
11. 最后运行五个通用 EvalScope benchmarks。

若 UltraFeedback 阶段无法证明 s 与受控遗漏/flip/discrepancy 有预期关系，应先修正 s，而不是继续投入昂贵的 RLCF/online experiments。

## 14. 主要来源

- TRL: https://github.com/huggingface/trl
- Original DPO code: https://github.com/eric-mitchell/direct-preference-optimization
- MMPO code: https://github.com/kykim0/margin-matching-pref-opt
- ODPO code: https://github.com/rycolab/odpo
- HelpSteer2: https://huggingface.co/datasets/nvidia/HelpSteer2
- NeMo-Aligner: https://github.com/NVIDIA/NeMo-Aligner
- RLCF: https://github.com/viswavi/RLCF
- WildChecklists: https://huggingface.co/datasets/viswavi/wildchecklists
- UltraFeedback: https://huggingface.co/datasets/openbmb/UltraFeedback
- UltraFeedback Binarized: https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized
- OpenRubric-v2: https://huggingface.co/datasets/OpenRubrics/OpenRubric-v2
- Auto-Rubric data: https://huggingface.co/datasets/agentscope-ai/Auto-Rubric
- Rubric-ARM paper: https://arxiv.org/html/2602.01511
- Rubric-ARM rubric model: https://huggingface.co/OpenRubrics/RubricARM-8B-Rubric
- Rubric-ARM judge: https://huggingface.co/OpenRubrics/RubricARM-8B-Judge
- Online Rubrics: https://arxiv.org/html/2510.07284
