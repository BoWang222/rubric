# EvalScope 通用能力 Benchmark 选择

> **状态：历史调研材料（非执行规范）。** 本文件保留当时的候选方案与术语，可能与最终设计不同。正式实验只以根目录 [`PLAN.md`](../../PLAN.md)、[`CHECKLIST.md`](../../CHECKLIST.md) 和 [`final_experiment_matrix_audit.md`](../scout/final_experiment_matrix_audit.md) 为准；尤其不得从本文沿用旧称 “Scaled DPO”、旧 scorer/gate 或旧数据映射。

更新时间：2026-08-08

## 推荐主表五项

| 能力 | Benchmark | EvalScope 名称 | 主指标 | 选择理由 |
|---|---|---|---|---|
| 综合知识 | MMLU | `mmlu` | Accuracy / macro average | 认知度最高、历史结果最多，适合与已有模型横向比较 |
| 数学推理 | GSM8K | `gsm8k` | Exact-match accuracy | 最常见的基础数学推理评测，运行稳定、成本低 |
| 复杂推理 | BBH | `bbh` | Average accuracy | 覆盖多种困难推理任务，和单纯知识选择题互补 |
| 代码生成 | HumanEval | `humaneval` | pass@1 | 最通用的代码能力标准评测；EvalScope 需要 sandbox 执行 |
| 专家级科学推理 | GPQA-Diamond | `gpqa_diamond` | Accuracy | 当前模型技术报告中常见，补充 MMLU 难度不足的问题 |

EvalScope 官方技能文档给出的 General LLM suite 是：

```text
mmlu gsm8k bbh humaneval ifeval
```

本研究已经在 rubric/instruction-following 主评测中使用 IFEval，因此通用能力表将 IFEval 替换为 `gpqa_diamond`，避免重复计算同一能力维度。

## 推荐运行原则

- 对所有训练方法使用同一 EvalScope 版本、dataset revision、prompt template 和 chat template。
- 统一确定性解码：`do_sample=false`、`temperature=0`；不要对不同方法单独调提示词。
- MMLU、GSM8K、BBH、GPQA-Diamond 使用 EvalScope 默认 metric 和官方 prompt；若修改 shot 数，所有模型固定相同配置并在论文中说明。
- HumanEval 正式报告 `pass@1`，使用 Docker sandbox；生成一个 completion 即可，避免把 sampling budget 当作模型提升。
- 主文同时报告训练前 SFT/reference checkpoint 与训练后模型，以显示 DPO/robust training 是否造成 capability regression。
- 对每个模型可给出五项原始分数，但不要直接平均不同 benchmark 的百分比作为唯一总分；若需要总体指标，报告相对 reference 的平均 normalized delta。

## 更难版本的可选替换

| 经典项 | 更难版本 | EvalScope 名称 | 使用建议 |
|---|---|---|---|
| MMLU | MMLU-Pro | `mmlu_pro` | 如果所有 7B/8B 模型在 MMLU 差距过小，可在附录补充或替换 |
| GSM8K | MATH-500 | `math_500` | GSM8K 饱和时补充，但不要为了增加表格同时堆叠大量数学集 |
| HumanEval | HumanEval+ | `humaneval_plus` | 更严格的隐藏测试；适合附录鲁棒性检查 |

主文优先保留经典版本，因为用户要求“最广为人知”，并且它们有最多可比较的历史结果。更难版本用于解决 saturation，而不是无条件替代经典 benchmark。

## 建议 EvalScope 调用

```bash
evalscope eval \
  --model YOUR_MODEL \
  --api-url OPENAI_API_COMPAT_URL \
  --api-key EMPTY_TOKEN \
  --datasets mmlu gsm8k bbh gpqa_diamond humaneval \
  --generation-config '{"do_sample":false,"temperature":0}' \
  --sandbox '{"enabled":true,"type":"docker"}'
```

正式实验前先对每个数据集运行 `--limit 5` 检查答案抽取和 sandbox，再移除 limit 完整评测。

## 来源

- EvalScope current supported LLM benchmarks: https://evalscope.readthedocs.io/en/v1.8.1/get_started/supported_dataset/llm.html
- EvalScope official general-suite recommendation and CLI discovery: https://github.com/modelscope/evalscope/blob/main/skills/evalscope/SKILL.md
- EvalScope HumanEval execution and sandbox requirements: https://evalscope.readthedocs.io/en/v1.8.1/benchmarks/humaneval.html
- EvalScope GPQA-Diamond: https://evalscope.readthedocs.io/en/v1.8.1/benchmarks/gpqa_diamond.html
