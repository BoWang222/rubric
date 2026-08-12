# ICLR27 Unmeasured Rubrics：实验执行清单

## Identity

- plan: `PLAN.md`
- stage: UltraFeedback baseline smoke/pilots verified; seed-42 DPO full run reached step 283 but failed while writing the first FSDP checkpoint; formal matrix paused for checkpoint-path repair
- main model: `Qwen/Qwen3-8B`（固定 revision；主轨 `enable_thinking=false`）
- current hardware contract: NUS shared node, at most 2×H100-80GB concurrently; FSDP1 FULL_SHARD; `per_device=1`, `gradient_accumulation=32`, no CPU offload, sequential jobs only

## 0. 2026-08-12 当前真实进度与下一执行边界

### 已验收，可直接复用

- [x] NUS `rubric` 环境、Qwen3-8B revision、UltraFeedback v2 processed/tokenized data、pair-keyed reference-logp cache 均已就绪
- [x] DPO/MMPO/ODPO/Scaled-DPO 公式对齐、source-parity tests、4条 1024-pair smoke 和5条 64-step pilot 通过
- [x] 只用 validation 冻结 `learning_rate=1e-6`、`ODPO alpha=0.5`；MMPO 依作者 8B recipe 固定 `gamma=2.2`
- [x] seed-42 阶段保持四 baseline 同 pairs/init/order/cache/budget；seeds `13/100` 仅作后续扩展

### 当前 blocker：正式 checkpoint 通路

- [x] DPO seed 42 已从 step 0 稳定训练到 step 283；约 `13.6--14.3s/step`，loss/gradient/throughput 在失败前均 finite
- [x] 失败点定位为 step-283 FSDP distributed-checkpoint save：rank 1 `NCCL Error 1: unhandled cuda error`
- [x] 该 `checkpoint-283` 仅有空目录（约 8KB）、无 trusted seal，不可恢复；当前无训练进程，`run_manifest.status=running` 是失败后的陈旧状态
- [ ] 让 launcher 在子进程失败时原子写 `failed` manifest，`status` 命令明确区分 active/stale/failed/incomplete-checkpoint
- [ ] 为正式 checkpoint 重试加入 NCCL/CUDA 诊断日志，并只做一次同配置 DPO seed-42 可比重试
- [x] DPO step-283 失败已定位为 PyTorch DCP 在 NCCL/CUDA 上执行 checkpoint planner 对象通信失败，不是 loss/数据/磁盘/超过两卡政策；改为独立 Gloo/CPU checkpoint metadata process group，训练梯度通信仍使用 NCCL
- [x] 修复后两卡 save+resume gate 通过：step 1 完整 FSDP model+optimizer+scheduler+RNG checkpoint 约 92GB，trusted seal 通过；从 checkpoint-1 恢复并训练/保存到 step 2 通过
- [x] launcher 仅恢复带 trusted seal 且 checksum 通过的 checkpoint；残缺 `checkpoint-283` 不再被误认为恢复点，异常后 `run_manifest` 会持久化 `failed`
- [ ] 仅重跑 DPO seed 42，确认真实 step-283 长训练 checkpoint 通过后，才放行后续 MMPO/ODPO/Scaled DPO
- [ ] 修复 gate 通过后从 DPO seed 42 重启；因当前无有效 checkpoint，必须从 step 0 重训

### 后续训练波次（严格顺序）

- [ ] Wave 1：完成 UltraFeedback seed-42 四 baseline，每条 566 steps + official-test 本地评测 + BF16 merge/finalize
- [ ] Wave 2：汇总四 baseline 的 test NLL/accuracy/reward margin/policy-reference KL/length；不用 test 反向改超参
- [ ] Wave 3：冻结 UltraFeedback 四维 canonical rubric 和 RubricARROW scorer，生成 500-prompt×16 response calibration bank，计算 `s_UF` + bootstrap CI
- [ ] Wave 4：先冻结 Wasserstein-W1 ground cost/rho normalization/solver 合同，再实现 KL 与 Wasserstein backends；通过 `rho=0`、`s=1`、explicit-inner 单测与 1k smoke
- [ ] Wave 5：在 validation 上只用 uniform control 选择一次全局 `rho`，然后运行 UltraFeedback nominal/uniform/KL-ours/W1-ours seed 42
- [ ] Wave 6：完成 WildChecklists/HelpSteer2 production data pipeline 与 baseline smoke；再开始其单-seed 训练
- [ ] Wave 7：只在原生与 automatic-rubric validity gates 通过后，才补 seeds `13/100` 和扩展 automatic-producer 矩阵

## A. 冻结研究合同

- [x] `s` 的单位是 exact criterion set：相同 rubric 共享 `s(C)`，不同 rubric 分别计算；prompt/producer 不属于 rubric 本体
- [x] 全局 rubric 可共享一个 `s`；prompt-specific generator 必须逐 rubric 计算，不能压成 system-level 常数
- [x] 不做 1D/2D/3D 或人工删维度
- [x] `s` 不修改 `m`，只乘在对应的无量纲 KL distance 前
- [x] `rho` 全实验固定
- [x] 不新增人工标注
- [x] MMPO 是核心 preference-label DRO consumer；ODPO 与预注册 Scaled-DPO normalized-gap transfer 使用 baseline-preserving outer loss-DRO extension；ML-RDPO 附录
- [x] native/structured-feedback systems：UltraFeedback、WildChecklists、HelpSteer2 human-magnitude anchor
- [x] automatic rubric producers：Auto-Rubric、OpenRubrics、Rubric-ARM、OnlineRubrics、RRD、EvoLM
- [x] 六个 producers 的主比较固定同一 WildChat prompts/candidate responses/common pointwise grader；OnlineRubrics-local 用独立 induction split 冻结 current/control policies，再逐 common prompt 用辅助 contrast rollouts 生成 rubric
- [x] 明确区分官方实现、发布资源复用、本地 adaptation 与后续 reimplementation
- [x] 每个 ours 行都必须有 `uniform robust (s=1)` 对照
- [x] 主矩阵表头是 rubric systems，不把 producer 名都称为 datasets；加入 HelpSteer2 human-magnitude anchor

## B. 环境与代码

- [x] 创建并冻结 conda environment
- [x] 记录 Python/PyTorch/CUDA/driver/package versions
- [x] 收集并记录 TRL 与 baseline repositories 的 exact commits
- [x] NUS 已同步九个只读 refs：OpenJudge `2151def`、RUBRIC-ARROW `d116811`、DPO `f8b8c0f`、EvoLM `207cf7e`、MMPO `ef3a91d`、ODPO `6152f67`、OpenRubrics `1a40c14`、RLCF `73254b6`、TRL `accf7383`
- [x] 记录 DPO/MMPO/ODPO 作者代码状态；HelpSteer2-Preference 含 Scaled DPO 公式与实验，但无独立官方 trainer；UF 行标 `Scaled-DPO normalized-gap transfer`
- [ ] 冻结 OpenRubrics、Rubric-ARM、EvoLM 官方代码/权重 commit
- [ ] 冻结 Auto-Rubric official data revision 与 RM-Gallery/OpenJudge 通用框架 revision；记录尚未验证该框架含论文完整 Propose--Evaluate--Revise/Theme--Tips pipeline，且无已确认的论文生成 checkpoint
- [ ] 冻结 EvoLM 仓的 RRD-WU reimplementation；记录其不是 RRD 原作者代码
- [ ] 保存 OnlineRubrics-local 的论文 prompt provenance 与 adaptation 说明
- [ ] 冻结 RubricARROW-8B-Judge 与官方 RUBRIC-ARROW code revisions、官方 JSON prompt、`top_logprobs=10` extractor、literal ` true/ false` log-prob 读取、`Z=clip((1+p_true-p_false)/2,0,1)` 与 failure contract
- [ ] 主 scorer 使用统一 RubricARROW contract；500 WildChecklists prompts只作跨方法 `s` 分布/parser/scorer diagnostic。正式 prompt-specific weights在各自 assignment prompts 上计算；criterion 缺失记 `Z=0`，producer parse success `<99%` 则不进主表
- [ ] 冻结 Prometheus-7B-v2 sensitivity revision、官方 absolute prompt、shared reference 与 exact 1--5 anchor strings/hash；只在固定 200 prompts 上运行，不进入主 pair/`m`/`s`
- [ ] 冻结 Qwen2.5-32B-Instruct shared-reference/extractor revision与 Qwen2.5-1.5B-Instruct RRD weak-reference revision
- [ ] 冻结 RRD-local 两个独立 actors：Qwen2.5-32B proposer 与 Qwen3-1.7B filter/item judge；保存 revisions、关闭-thinking设置、decoding 与 `rrd_wu` overlay
- [x] 冻结 `Qwen/Qwen3-8B` exact revision、官方 chat template 与 `enable_thinking=false`；确认 `transformers>=4.51.0` 并保存 tokenizer/template hash
- [x] 在 4×A800 上完成 Qwen3-8B BF16 full-parameter DPO/MMPO/ODPO/Scaled-DPO smoke：FSDP1 FULL_SHARD、FlashAttention 2、gradient checkpointing、durable reference log-prob cache、`max_length=2048`；两卡探测观测约 81.1GB 并按预注册 65GB 门槛回退四卡顺序
- [ ] LoRA 仅作代码 smoke/可选参数高效 ablation；若 full-parameter smoke 失败，先记录失败配置与显存，再决定是否将整个比较组统一降级，禁止 baseline/ours 混用 full 与 LoRA
- [x] 创建 configs/data/cache/artifacts 目录
- [x] 实现并通过统一数据 schema/manifests 全量 gate
- [x] 生成 pair-keyed reference-logp cache，并通过 cached-vs-online 严格对齐 gate

## C. Loss 与数值正确性

- [x] Vanilla DPO 与 TRL 公式 FP64 对齐，GPU cached-vs-online reference-logp gate 已通过
- [x] MMPO 与作者公式/实现 FP64 对齐
- [x] ODPO author log-gap 公式对齐且 `alpha=0` 退化测试通过
- [x] Scaled-DPO normalized-gap transfer unit-weight 与 native 1:2:3 退化测试通过
- [ ] `rho==0` 显式代码分支：MMPO 返回共享同一 `P0` 的 stabilized nominal loss；ODPO / Scaled-DPO normalized-gap transfer 返回各自冻结的 nominal loss
- [ ] 对 `rho->0+` 检查两类 dual 的数值极限；不要求有限 `eta` 在 `rho=0` 取得 nominal value
- [x] 记录 UltraFeedback MMPO target clipping count 与 p0 分布
- [ ] 每个 consumer 的 ours at `s=1` 与其 uniform robust 完全一致
- [ ] chosen/rejected branch symmetry 通过
- [ ] MMPO per-rubric weighted dual 与显式 inner maximization 小规模对齐
- [ ] ODPO/adapter per-rubric weighted generalized-KL dual及一维 normalization root 与显式 inner solver 对齐；constant-`s` 时退化为 scalar-`s` 公式
- [ ] ODPO/adapter outer extension 固定 `B_DRO=8`，inner batch同 system；gradient accumulation 后 effective batch仍为64
- [ ] 验证 label KL 除以 `-log(epsilon)`、minibatch loss-distribution KL 除以 `log(B_DRO)`，二者数值均落在 `[0,1]`
- [ ] 确认只有一个全局 `rho`；不创建 `rho_label/rho_loss`
- [ ] 确认 `s` 不进入 `m`、原 baseline sample weight 或 `rho`；只进入 distance coefficient，并可由 inner maximization间接改变 adversarial `Q`

## D. 数据准备

- [x] 下载并冻结 raw UltraFeedback revision；验证 63,967 prompts / 255,864 completions 且包含 2023-12-29 overall-score fix
- [x] 下载并冻结 `HuggingFaceH4/ultrafeedback_binarized` revision；验证 train_prefs=61,135、test_prefs=2,000 与 `prompt_id`
- [x] 对齐 UltraFeedback pair 与 4-aspect scores；最终 train/validation/test=36,258/2,000/1,228
- [x] 验证 raw revision 已含官方 2023-12-29 `overall_score` 修复；four-aspect 重算、overall audit-only、train-only q95=3.5
- [x] 下载并验证 WildChecklists revision `f4175828...`：51,071 rows，schema 为 `prompt/chosen/rejected/chosen_score/rejected_score/requirements`，无 criterion-score 列
- [ ] 为 WildChecklists 冻结 calibration/train/validation/test IDs、aggregate-gap normalization、Qwen3 tokens 与 reference-logp cache
- [ ] 若用 RLCF 仓另行链接的 criterion-level/scoring intermediates，单独冻结 revision/checksum
- [x] 下载并验证 HelpSteer2 revision `990b2711...`：21,362 ratings（20,324/1,038）；preference 9,125 rows，去除 2,007 zero-strength 后为 6,766/352，strength 仅 `-3..3`
- [ ] 冻结 HelpSteer2 完整五维 guidance/hash，生成 structured-feedback 主轨 processed splits/tokens/reference cache
- [ ] 从 WildChecklists 官方 release 按固定 hash 先留出 500 prompts 作九-system common calibration，并从政策训练中移除
- [x] 下载 WildChat-1M revision `7d6490e4...` 全14个 parquet shards；尚未生成项目 splits
- [ ] 固定 WildChat 4k train / 500 val + 500 OnlineRubrics induction split；不再另造 WildChat calibration；扩容只能在全量训练前统一预注册
- [x] 下载 Auto-Rubric revision `2a5e18a3...` official data（38,459 rows）与 OpenRubric-v2 revision `d1048e9e...`（74,214 rows）
- [ ] 完成 Auto-Rubric/OpenRubric-v2 method-native schema/provenance checks；不将它们直接当成 common controlled training corpus
- [ ] 从 Auto-Rubric 论文 Appendix K 转录并双人/双解析校验 HelpSteer3-source global Theme--Tips rubric
- [ ] 记录 OnlineRubrics Generalist/Expert data unavailable；不得创建虚假的原始 dataset manifest
- [ ] 冻结 Tulu-3 preference-mixture revision 作为 EvoLM provenance；不要求重训完整 co-evolution
- [x] 下载 JudgmentBench revision `945ff52f...`：30 tasks、457 rubric items、2,274 outputs、1,530 comparative judgments、1,539 rubric annotations
- [ ] 验证 JudgmentBench expert rubric/comparative-judgment joins 与去重；仅作 independent diagnostic
- [ ] 对全部 splits 和 evaluation prompts 做去重
- [x] 保存 UltraFeedback 数据过滤、quarantine、split 与 margin normalization manifests
- [x] UltraFeedback 只用 train raw positive gaps 的 NumPy `quantile(method="linear")` 冻结 `q95_g=3.5`；val/test 复用 train q95 与 `mu_train`
- [x] UltraFeedback `q95_g>0`、raw-tie、near-tie、clipping 与最终 36,258/2,000/1,228 pair counts 全部通过

## E. `s` calibration

- [ ] 定义 canonical rubric JSON/hash：只含 criteria 及绑定 guidance/anchors/direction/importance，不含 prompt、response、producer、generator 或 grader
- [ ] 验证相同 canonical rubric 共享同一 `rubric_id -> s` cache；不同 rubric 不因 producer 相同而合并
- [ ] pilot 后全局冻结 `M`（主配置 16）；为实际 rubric assignments 生成/复用 probe response bank，报告 `K_r>M-1` 比例并做 `M=8/32` sensitivity
- [ ] 为 WildChecklists calibration、WildChat common/induction prompts 生成共享 Qwen2.5-32B reference bank；与 candidate/Online auxiliary responses 分离
- [ ] 生成 RRD Qwen2.5-1.5B weak-reference bank；shared reference 复用为 strong bank；两者不进入 candidate pool
- [ ] 冻结 criterion grader 与 parsing schema
- [ ] 全局 rubric：逐 prompt 算 covariance/log-det 后在共同 calibration prompts 平均成一个 `s`
- [ ] prompt-specific rubric：在其来源 prompt 的全部 probe responses 上计算自己的 `s(C_i)`；exact rubric 在多个 prompts 重复时才在这些 assignment prompts 内平均
- [ ] prompt-specific `s` 在 top/bottom pair 选择前由全部 probe score vectors计算，不使用最终 chosen/rejected label
- [ ] 所有 criterion scores 先变成“越高越好”，ROIV importance 必须非负；HelpSteer2 使用 `v_minus=1-v/4` 与权重 `0.40`
- [ ] 计算 `s_UF`
- [ ] WildChecklists：对每个 exact `requirements` checklist 计算/cache `s(C_i)`，不得生成一个 `s_WC` 常数
- [ ] 计算 `s_HS2`
- [ ] 用 Auto-Rubric paper global rubric + common scorer 计算 `s_AR`
- [ ] OpenRubrics：逐 generated rubric 计算 `s(C_i)`
- [ ] Rubric-ARM：逐 generated rubric 计算 `s(C_i)`
- [ ] 用冻结 Qwen2.5-7B 为独立 induction split 生成 `M_induction=8` candidate bank（temp 0.7/top-p 0.9/独立 seed+hash），冻结 OnlineRubrics-local current/control；对每个 prompt 的 `dedup(C0 union Ce)` 分别计算 `s(C_i)`
- [ ] RRD-local：逐 recursive rubric 计算 `s(C_i)`；common/native judge 两轨分开
- [ ] EvoLM：逐 generated JSON rubric 计算 `s(C_i)`
- [ ] 全局 rubric 做 prompt bootstrap；unique prompt-specific rubric 做 response bootstrap/独立 probe-seed stability；保存 producer-level分布与分位数
- [ ] 检查 split-half 与 response-seed stability
- [ ] 小规模评估 judge-noise subtraction
- [ ] 在 HelpSteer2-Preference、JudgmentBench 与已有 human-preference evaluation 上做独立 agreement diagnostic
- [ ] Gate 1 通过或记录 claim downgrade

## F. UltraFeedback native 主实验

- [x] 1024-pair baseline smoke：DPO/MMPO/ODPO/Scaled-DPO 全部 16 steps 通过；DPO step-8 resume 通过；状态 `verified-for-controlled-UF`
- [x] 64-step development pilots：5/5 完成，只用 validation 冻结 `learning_rate=1e-6`、`ODPO alpha=0.5`；MMPO gamma 按作者8B UltraFeedback recipe固定为 `2.2`
- [x] NUS 服务器合规双卡 pilot 配置：`per_device=1`、`gradient_accumulation=32`、effective batch `64`、FSDP no-offload；8-step实测 `13.7s/step`、peak reserved `77.37GB/GPU`。`per_device=2` + CPU offload 实测更慢（`67.1s/step`），不采用
- [x] NUS 双卡 no-offload checkpoint gate：step 1 保存、恢复到 step 2、最终 BF16 合并与 recovery cleanup 均通过；单 checkpoint 约 `92GB`，合并后约 `16GB`
- [ ] 第一阶段：seed `42` 上的四 baseline，每条 566-step 全量 UltraFeedback 训练与 official-test 本地评测
- [ ] DPO seed 42：第一次尝试在 step 283 checkpoint 失败；修复 checkpoint 通路后从 step 0 重启
- [ ] MMPO seed 42：未启动
- [ ] ODPO seed 42：未启动
- [ ] Scaled-DPO normalized-gap transfer seed 42：未启动
- [ ] 第二阶段：资源允许时补 seeds `13/100`，升级为预注册的三 seed 主表；单 seed 结果不报种子方差
- [ ] Gate 0 通过后，在 UltraFeedback development split 仅用 uniform robust 网格选择一次全局 `rho`；立即冻结，ours/其他数据/consumer 不重调
- [ ] 单 seed：Vanilla DPO
- [ ] 单 seed：MMPO nominal/uniform/ours
- [ ] 单 seed：ODPO nominal / loss-DRO uniform / loss-DRO ours
- [ ] 单 seed：Scaled-DPO normalized-gap transfer nominal / loss-DRO uniform / loss-DRO ours
- [ ] 单 seed：ML-RDPO appendix
- [ ] 检查训练 diagnostics 与 baseline comparability
- [ ] DPO 与 MMPO nominal/uniform/ours 提升到 seeds 13/42/100；ODPO / Scaled-DPO normalized-gap transfer 仅按四-system transfer 合同升 seeds
- [ ] 单系统 oracle diagnostic：仅用 UltraFeedback dev 从同一预注册网格选择 uniform-robust radius，3 seeds 报告其与 fixed-`rho` ours 的差异；不得回流修改全局 `rho` 或 ours

## G. WildChecklists native 主实验

- [ ] 在统一 trainer 中运行 WildChecklists released-data + standard DPO baseline；未验证中间产物前不称 full-pipeline reproduction
- [ ] 直接构造并冻结 released aggregate gap `m_raw_i=chosen_score-rejected_score`，再应用统一 train-only q95 normalization；只有取得中间产物后才做 criterion-level reconstruction
- [ ] 单 seed：MMPO nominal/uniform/ours
- [ ] 单 seed：ODPO nominal / loss-DRO uniform / loss-DRO ours
- [ ] 单 seed：Scaled-DPO normalized-gap transfer nominal / loss-DRO uniform / loss-DRO ours
- [ ] 检查训练 diagnostics 与 baseline comparability
- [ ] DPO 与 MMPO nominal/uniform/ours 提升到 seeds 13/42/100；ODPO / Scaled-DPO normalized-gap transfer 仅按四-system transfer 合同升 seeds
- [ ] WildChecklists rubric-level diagnostic：比较 correct `rubric_id -> s`、within-producer shuffled、rank-reversed 与 uniform；可另报 validation-tuned uniform radius，但不得声称其与 prompt-specific ours 数学等价

## H. HelpSteer2 structured-feedback adaptation 主实验

- [ ] 用 `0.65h+0.80c+0.45coh+0.55comp+0.40(4-v)` 构造分数；显式令 `chosen=argmax r`、`rejected=argmin r`、`m_raw=r_chosen-r_rejected`，再应用统一 train-only q95 normalization；该 raw gap 与原 `-0.40v` 等价
- [ ] sensitivity：只用 `0.65h+0.80c+0.45coh`；不得用结果选择主聚合
- [ ] 验证 rubric direction 与独立 human preference direction
- [ ] 单 seed：Vanilla DPO
- [ ] 单 seed：MMPO nominal/uniform/ours
- [ ] 单 seed：ODPO nominal / loss-DRO uniform / loss-DRO ours
- [ ] 单 seed：Scaled-DPO normalized-gap transfer nominal / loss-DRO uniform / loss-DRO ours
- [ ] appendix：human-strength-weighted policy-loss adapter；明确来自 Scaled BT 的 DPO adaptation，不是 paper reproduction
- [ ] DPO 与 MMPO nominal/uniform/ours 提升到 seeds 13/42/100；ODPO / Scaled-DPO normalized-gap transfer 仅按四-system transfer 合同升 seeds
- [ ] 单系统 oracle diagnostic：仅用 HelpSteer2 validation 从同一预注册网格选择 uniform-robust radius，3 seeds 报告其与 fixed-`rho` ours 的差异；不得回流修改全局 `rho` 或 ours

## I. Automatic-rubric 主实验

- [ ] 对 WildChat 4k train / 500 val prompts 一次性采样全局冻结的 `M` 个 common responses（主配置 16）；所有 producers 复用
- [ ] OnlineRubrics-local induction：用论文 synthetic-offline-rubric prompt 与冻结 Qwen2.5-32B extractor 生成 `C0`，在独立 500-prompt split 做一轮 nominal-MMPO warm-up；冻结 `Pi_current`、初始 `Pi_control`、extractor、decoding、warm-up steps 与 dedup config
- [ ] OnlineRubrics-local common/calibration prompts：逐 prompt 生成 `C0(x)`；current/control 各采 8 个 rollouts并按 sample index 构 8 对，得到 `Ce(x)`；冻结 `dedup(C0 union Ce)` 后给共享 candidate bank 评分
- [ ] 禁止跨 prompt 套用 induction rubric，禁止把 Online auxiliary rollouts混入 candidate comparison
- [ ] Auto-Rubric：common bank 只接入论文 Appendix K 的 HelpSteer3-source global rubric；official query-specific data 另做 artifact check，禁止跨 prompt 套用
- [ ] OpenRubrics：运行官方 RubricRM-v2 local generator
- [ ] Rubric-ARM：运行官方 released generator
- [ ] EvoLM：运行官方 frozen EvoLM-8B generator
- [ ] RRD：运行 EvoLM 仓 RRD-WU reimplementation，并标记非 RRD 官方 code；用冻结 strong/weak references真正启用 misalignment filter
- [ ] RRD main：用 RubricARROW `1[p_true>=p_false]` 形成每个 recursive rubric 的 binary matrix并重算 WU，保存 rubric-level `s_RRD_common(C_i)`；native judge轨另存，禁止混用
- [ ] 所有 producer 的实验标签逐项检查：official reproduction / official-artifact transfer / local adaptation / later reimplementation
- [ ] 用同一 RubricARROW official prompt/extractor 得到 criterion scores；500 WildChecklists prompts用于诊断，正式 `s(C_i)` 使用每个 rubric 的 assignment prompt response bank；固定 parser/failure rule
- [ ] 固定同一 200-prompt 子集，用 Prometheus-7B-v2 + shared reference + exact anchors 做 scorer sensitivity；不得用 sensitivity 结果改主配置
- [ ] 用统一 top/bottom、tie filter 和 train-only quantile normalization 生成 winner/loser/margin
- [ ] 六个 producers 的 parser success/non-tie/common-grader validity 通过 Gate 3
- [ ] method-native artifact checks 与 common controlled data 分目录、分表保存；禁止跨 native data 做 producer 排名
- [ ] 六个 producers 各跑 producer-specific Vanilla DPO（同 pair、忽略 `m`）
- [ ] Auto-Rubric：MMPO nominal/uniform/ours
- [ ] OpenRubrics：MMPO nominal/uniform/ours
- [ ] Rubric-ARM：MMPO nominal/uniform/ours
- [ ] OnlineRubrics-local：MMPO nominal/uniform/ours
- [ ] RRD-local：MMPO nominal/uniform/ours
- [ ] frozen EvoLM：MMPO nominal/uniform/ours
- [ ] 构造六个自然完整 rubric systems 的等量 joint mixture，`pi_g=1/6`
- [ ] joint MMPO nominal
- [ ] joint MMPO + uniform robust
- [ ] joint MMPO + ours
- [ ] joint MMPO + within-producer shuffled `rubric_id -> s`（重复 rubric 一起移动）
- [ ] joint MMPO + within-producer rank-reversed rubric-level `s`（保留各 producer 的 multiset/均值，不用 `1/s`）
- [ ] optional：cross-producer rubric-level shuffle，匹配样本数后保留全局 multiset
- [ ] joint nominal/uniform/ours/shuffled/rank-reversed 全部提升到 seeds 13/42/100
- [ ] 九个 rubric systems：ODPO nominal / loss-DRO uniform / loss-DRO ours 单 seed，作为 coverage appendix
- [ ] 九个 rubric systems：Scaled-DPO normalized-gap transfer nominal / loss-DRO uniform / loss-DRO ours 单 seed，作为 coverage appendix
- [ ] 预注册 UltraFeedback、WildChecklists、Rubric-ARM、frozen EvoLM 的 ODPO / Scaled-DPO normalized-gap transfer extensions 升 3 seeds
- [ ] 九-system 核心主表的 producer-specific DPO、MMPO nominal/uniform/ours 全部提升到 seeds 13/42/100
- [ ] optional：每个需闭源模型的方法仅做 200-prompt local-vs-faithful sensitivity
- [ ] 表脚明确：Auto-Rubric=global rubric；OnlineRubrics=local frozen DPO adaptation；RRD=EvoLM authors' reimplementation；EvoLM=frozen official-checkpoint transfer

## J. Ablations

- [ ] uniform `s=1`
- [ ] criterion-count proxy
- [ ] text-embedding proxy
- [ ] within-producer rubric-level shuffled `s(C)`；global rubric 不伪造 within-system shuffle
- [ ] within-producer rubric-level rank-reversed `s(C)`（保留原权重 multiset/均值）
- [ ] judge-noise subtraction
- [ ] `L/M` sensitivity
- [ ] global lambda sensitivity
- [ ] globally fixed rho sensitivity
- [ ] KL vs chi-square appendix
- [ ] probe/grader small-scale stability

## K. Evaluation

- [ ] held-out preference NLL/accuracy/AUC/calibration
- [ ] JudgmentBench：同输出上的 expert rubric-score vs expert pairwise diagnostic
- [ ] 全局 rubric：`s`/CI/effective radius；prompt-specific：rubric-level `s(C_i)` 分布、response-seed stability 与 adversarial shift-by-`s`
- [ ] policy KL/reward margin/length/gradient/dual diagnostics
- [ ] AlpacaEval2-LC
- [ ] Arena-Hard-SC
- [ ] WildBench
- [ ] IFEval
- [ ] InfoBench
- [ ] FollowBench for WildChecklists track
- [ ] 3-seed statistics and paired bootstrap
- [ ] position/order-bias evaluation

## L. OnlineRubrics-local extension

- [ ] offline core evidence passes gates
- [ ] implement 2--3 round IterDPO adaptation
- [ ] 明确实验名为 `OnlineRubrics-local DPO adaptation`，不写成原文 GRPO 复现
- [ ] 每轮 current policy 与固定 control policy 分别采样；保存并核对两套 rollout provenance
- [ ] fixed rubric control
- [ ] rubric update only
- [ ] online + uniform robust
- [ ] online + ours
- [ ] verify one `s_t(C_t(x))` per exact rubric per round；相同 rubric 共享 cache，`rho` 始终固定

## M. 最终通用 EvalScope benchmarks

- [ ] freeze checkpoints before general evaluation
- [ ] MMLU
- [ ] GSM8K
- [ ] BBH
- [ ] HumanEval
- [ ] GPQA-Diamond
- [ ] 汇总能力 regression 与平均 response length

## N. Closeout

- [ ] 每个 run 都有 config、command、log、manifest、metrics 与 checkpoint
- [ ] 主张逐项标为 supported/refuted/inconclusive
- [ ] 记录所有失败与不可比较结果
- [ ] 完成主表、消融表、数据统计表和计算成本表
- [ ] 明确下一步：继续实验、修改方法或开始写作
