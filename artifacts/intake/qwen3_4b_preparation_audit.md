# Qwen3-4B DPO 实验准备审计

> **状态：历史调研材料（非执行规范）。** 本文件保留当时的候选方案与术语，可能与最终设计不同。正式实验只以根目录 [`PLAN.md`](../../PLAN.md)、[`CHECKLIST.md`](../../CHECKLIST.md) 和 [`final_experiment_matrix_audit.md`](../scout/final_experiment_matrix_audit.md) 为准；尤其不得从本文沿用旧称 “Scaled DPO”、旧 scorer/gate 或旧数据映射。

> **主模型决策已更新：** 论文主表固定为 `Qwen/Qwen3-8B` BF16 full-parameter fine-tuning；下述 Qwen3-4B LoRA 结果仅是历史 smoke evidence，不能迁移成主模型或正式训练配置。

更新时间：2026-08-10

## 结论

当前状态为 `baseline_partial / usable_with_verification`：Qwen3-4B 的单卡加载、生成、preference pair 前向、TRL DPOTrainer、LoRA 保存链路均已跑通；但这只证明 vanilla DPO-LoRA smoke 可执行，尚未达到正式实验启动门槛。

## 已确认、可复用的状态

- 服务器项目根目录：`/home/chunyuan/rubric`
- Conda 环境：`rubric`，Python 3.11
- GPU：NVIDIA H100 80GB；成功运行使用物理 GPU 1，在进程内映射为 `cuda:0`
- 模型：`Qwen/Qwen3-4B`
- 模型 revision：`1cfa9a7208912126459214e8b04321603b3df60c`
- 训练栈：
  - PyTorch `2.5.1+cu121`
  - Transformers `4.52.4`
  - TRL `0.19.1`
  - PEFT `0.15.2`
  - Datasets `3.6.0`
  - Accelerate `1.8.1`
  - DeepSpeed `0.17.2`
- 依赖检查：`python -m pip check` 通过
- UltraFeedback smoke 数据：`/home/chunyuan/rubric/data/processed/ultrafeedback_smoke_1k`
- 数据内容 SHA256：`9a7a5461c509986de432209029f98cfc74200f66a5392a7e7aabaeeaaab73a4f`
- 完整扫描统计：61,135 pairs；7,387 ties；53,748 positive-margin pairs
- 成功运行：`dpo_lora_qwen3_4b_smoke20_seed42_r4`
  - 数据：smoke 数据前 128 条
  - steps：20
  - seed：42
  - effective batch：4
  - max length：2048
  - beta：0.1
  - LR：`5e-6`
  - train loss：`0.7092550724744797`
  - runtime：`41.933s`
  - peak allocated GPU memory：`13.458 GiB`
- adapter、tokenizer、trainer state、manifest、metrics、TensorBoard event 均已保存
- adapter reload gate：已从固定 base revision 与 `r4/final_adapter` 独立重载并完成确定性生成

## Trust ranking

| 资产 | 判断 | 说明 |
|---|---|---|
| CUDA/Qwen3 BF16 加载和生成 | trusted | 已在单张 H100 上实际通过 |
| chosen/rejected 前向 | trusted | logits finite，长度 512 的 pair batch 通过 |
| vanilla DPO-LoRA 20-step smoke | trusted as execution smoke | 能执行和保存，但不是效果证据 |
| UltraFeedback smoke 1k | usable_with_verification | 内容已冻结；当前训练脚本只使用前 128 条，manifest 中的 available rows 因此记为 128 |
| 53,748 positive-margin full set | usable_with_verification | 计数已确认，但正式 split/schema/revision manifest 尚未完成 |
| Qwen3-4B 作为正式主模型 | reference_only / superseded | 当前方法计划中的主 controlled model 已冻结为 `Qwen/Qwen3-8B`；Qwen3-4B 只保留为历史开发 smoke，不进入论文主表 |
| 本文 ROIV-conditioned robust objective | missing implementation | 当前只跑通 vanilla DPO，尚无 MMPO/ODPO/Scaled/uniform/ours loss 实现与等价性单测 |
| adapter 独立重载 | trusted | 固定 base revision 加载后成功挂载 r4 adapter 并生成 |
| 正式评测闭环 | missing | 尚未完成验证集指标、Base/adapter 对照生成和 benchmark 管线 |

## 正式实验前必须完成

1. 简化运行接口：模型、revision、默认 GPU 和自动 run id 进入脚本默认值或 YAML config；常规运行不能依赖多条手工 `export`。
2. 数据合同：冻结 full train/validation/test，记录 dataset repository、revision、config、official split、过滤规则、样本数、fingerprint 与内容 hash。
3. 正式训练脚本与配置：smoke 和正式 run 分离；正式 run 要有 eval/save strategy、resume、失败状态、resolved config 和完整日志。
4. loss 正确性单测：至少完成 DPO、MMPO、ODPO、Scaled DPO、uniform robust、ours，以及计划中列出的退化/对称性测试。
5. 评测 smoke：Base 与训练后 adapter 使用同一 prompts 和 generation config，验证生成、长度和最低限度 preference/ranking metric 管线。
6. 资源合同：明确 Qwen3-4B 阶段是 LoRA pilot 还是 full-parameter pilot；两张 H100 如何分配不能静默变化。

## 可在 pilot 后补，但必须在论文主实验前完成

- 三 seeds 和置信区间报告脚本
- specialized benchmarks 的统一入口
- full-parameter DPO 或与主表一致的参数高效训练决策
- Qwen3-8B 正式全参合同与 Qwen3-4B 历史 LoRA smoke 之间的隔离说明
- 代码版本控制、commit/patch provenance 与独立 run branch

## 下一锚点

继续 `experiment` 准备阶段，不启动 full training。adapter reload gate 已通过；下一步依次完成：full dataset contract -> 正式训练配置 -> loss 单测骨架 -> evaluation smoke。全部通过后再生成最终正式实验交接文档。
