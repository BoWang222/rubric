# Rubric-aware DPO 数据集可行性审计

> **状态：历史调研材料（非执行规范）。** 本文件保留当时的候选方案与术语，可能与最终设计不同。正式实验只以根目录 [`PLAN.md`](../../PLAN.md)、[`CHECKLIST.md`](../../CHECKLIST.md) 和 [`final_experiment_matrix_audit.md`](../scout/final_experiment_matrix_audit.md) 为准；尤其不得从本文沿用旧称 “Scaled DPO”、旧 scorer/gate 或旧数据映射。

## 结论

若要求数据集或官方流程能够为同一 prompt 的多个回答提供逐 criterion 分数，并可用于 DPO，当前文本任务中最可行的组合是：

1. **原始 UltraFeedback**：最适合低成本验证固定 rubric 的数据集级标量 \(s(\mathcal C)\)。原始数据每题已有 4 个回答及四个维度的分数；严格实验再用同一冻结策略重采样 \(M\) 个回答即可。
2. **WildChecklists + RLCF 官方代码**：最适合作为真正 rubric-grounded DPO 的主实验或迁移实验。发布的 DPO 表只有 chosen/rejected，但官方生成与评分管线可以扩展到 \(M\) 个回答并逐 criterion 评分。

若严格要求“静态公开数据里已经同时包含同一策略生成的 \(M\) 个回答、逐 criterion 分数和 DPO 对”，目前没有成熟文本数据集完全满足。合理做法是保留既有 DPO 数据不变，另在一小部分 prompt 上建立只用于估计 \(s(\mathcal C)\) 的 calibration sidecar。

## 可行性矩阵

| 数据集/方法 | 可直接构造 DPO | 显式 rubric | 已发布的每题回答数 | 逐 criterion 分数 | 能否按原流程评价新回答 | 结论 |
|---|---:|---|---:|---:|---:|---|
| [UltraFeedback 原始版](https://huggingface.co/datasets/openbmb/UltraFeedback) | 是 | 固定四维 | 4 | 是 | 是，已有公开评分提示/任务实现 | **首选** |
| [WildChecklists / RLCF](https://github.com/viswavi/RLCF) | 是 | 每题 checklist | 发布表为 2；代码可改为 \(M\) | 官方流程支持 | 是 | **强候选，但昂贵** |
| [Visual rDPO](https://arxiv.org/html/2604.13029) | 是 | 每题 checklist | 32 | 是 | 论文流程支持 | 方法匹配度高，但公开代码/数据状态不清，暂不作为首选 |
| [RubricHub v1](https://huggingface.co/datasets/sojuL/RubricHub_v1) | 否，原文主要是 RuFT/RuRL | 每题 rubric | 论文候选池为 6 | 是 | 是 | 适合辅助验证 \(s\)，不是成熟 DPO baseline |
| [OpenRubrics](https://huggingface.co/datasets/OpenRubrics/OpenRubrics) | 是，pairwise 格式 | 每题 rubric | 2 | 否，主要是成对胜负 | 只能按 pairwise judge | 不适合估计回答分数向量的协方差 |
| HelpSteer2 / HelpSteer2-Preference | 是 | 固定五维 | 2 | 有人工分数 | 无需人工时不能原样评价新回答 | 不适合当前约束 |

## 推荐设置一：UltraFeedback

定义固定 rubric：

\[
\mathcal C_{\mathrm{UF}}=\{\text{instruction-following, truthfulness, honesty, helpfulness}\}.
\]

官方原始数据包含约 64k 条指令、每条 4 个回答，以及每个回答在四个方面的评分与解释。它由同一题目的多个候选自然形成 DPO 对，也是成熟的 UltraFeedback-DPO 系列实验的数据源。

### 低成本可行性实验

- 直接读取原始 UltraFeedback 的 4 个回答和四维评分。
- 对每个 prompt 内的分数向量先中心化，再跨 prompt 汇总，减少题目难度造成的伪协方差。
- 得到一个固定的 \(s(\mathcal C_{\mathrm{UF}})\)，对该数据集全部 DPO 样本共用。

这一版本可证明代码和统计量可以运行，但四个回答来自随机模型池，并非同一冻结策略，因此不应作为最严格的主结果。

### 严格主实验

- 从数据中抽取 \(L=1{,}000\) 个 calibration prompts；无需对全量数据重新评分。
- 使用一个冻结的 probe policy，对每题生成 \(M=8\) 个回答，固定温度和采样参数。
- 使用一个冻结 judge，按 UltraFeedback 的完整评分说明为每个回答输出四维分数 \(z_{\ell j}\)。rubric 应包含完整评分指令，不只是四个标签。
- 使用 prompt 内协方差：

\[
\widehat\Sigma_{\mathrm{within}}
=\frac1L\sum_{\ell=1}^{L}
\operatorname{Cov}_{j=1,\ldots,M}(z_{\ell j}).
\]

- 再按既定信息量统计量映射为一个数据集级 \(s(\mathcal C_{\mathrm{UF}})\)。
- DPO 训练仍可使用成熟的 UltraFeedback binarized 对；calibration sidecar 只负责估计一次 \(s\)，不把 \(s\) 做成样本函数。

## 推荐设置二：WildChecklists / RLCF

RLCF 的公开数据包含 prompt、chosen、rejected、两者总分和 checklist requirements；官方代码同时提供候选生成、逐 requirement 评分和 DPO 训练流程。其 README 中的候选生成数量原本为 2，可改为 \(M=8\)。评分器以 0--100 评价每个回答对每条 requirement 的满足程度。

建议：

- DPO 主训练仍使用发布的约 51k preference pairs。
- 仅抽取 \(L=500\) 至 \(1{,}000\) 个 prompt 估计一次 \(s\)。
- 每题由同一冻结 policy 生成 \(M=8\) 个回答。
- 使用官方 rubric-grounded judge；成本受限时每项重复 3--5 次，而非论文最昂贵的 25 次。
- 因 checklist 的维数和内容随题目变化，先计算每题的信息统计量，再平均：

\[
\widehat{\mathcal I}_{\mathrm{RLCF}}
=\frac1L\sum_{\ell=1}^{L}\widehat{\mathcal I}(\mathcal C_\ell),
\qquad
s(\mathcal C_{\mathrm{RLCF}})=g(\widehat{\mathcal I}_{\mathrm{RLCF}}).
\]

最终仍只有一个系统/数据集级 \(s\)。不过它衡量的是“checklist 生成器 + 评分协议”整体，而不是一段固定 rubric 文本，因此理论叙述不如 UltraFeedback 干净。

## 不建议作为首个主实验的原因

- **OpenRubrics**：发布的是 response A/B 和 winner，pairwise judge 不产生每个回答的独立逐维数值向量，无法直接计算当前的信息统计量。
- **HelpSteer2**：逐维分数来自人工；新增 \(M\) 个回答后无法在“不增加人工标注”的约束下沿用相同测量机制。
- **RubricHub**：评分结构很好，但原论文的优化路径不是标准 DPO；自行把 top/bottom 变成 DPO 会增加一个自建 baseline 变量。
- **Visual rDPO**：论文设置几乎完全匹配，但目前未确认有可直接复现的公开数据和代码，而且是多模态，工程成本明显更高。

## 最终选择

论文主线可采用两层实验：

1. **UltraFeedback**：验证固定 rubric 的 \(s(\mathcal C)\) 定义、统计稳定性和 robust DPO 增益。这是最干净的主实验。
2. **RLCF/WildChecklists**：验证方法能增量接入已有 rubric-grounded DPO 管线。这是更强但更昂贵的外部有效性实验。

关键点是：估计 \(s\) 不要求重建全量偏好数据。只需在 calibration prompts 上生成并评分 \(M\) 个回答；得到的 \(s\) 随后固定地用于该数据集的全部训练。

## 主要来源

- [UltraFeedback 官方数据卡](https://huggingface.co/datasets/openbmb/UltraFeedback)
- [UltraFeedback 官方仓库](https://github.com/OpenBMB/UltraFeedback)
- [TRL 的 UltraFeedback 处理脚本](https://github.com/huggingface/trl/blob/main/examples/datasets/ultrafeedback.py)
- [RLCF 官方仓库](https://github.com/viswavi/RLCF)
- [WildChecklists 官方数据集](https://huggingface.co/datasets/viswavi/wildchecklists)
- [RLCF 论文](https://arxiv.org/html/2507.18624)
- [Visual rDPO 论文](https://arxiv.org/html/2604.13029)
- [RubricHub v1 数据集](https://huggingface.co/datasets/sojuL/RubricHub_v1)
- [OpenRubrics 官方数据集](https://huggingface.co/datasets/OpenRubrics/OpenRubrics)
