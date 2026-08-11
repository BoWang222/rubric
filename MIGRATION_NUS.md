# Migration to `nus-container`

## Audited target

- Host account: `/home/chunyuan`
- GPUs currently visible: 8 x NVIDIA H100 80GB HBM3 with NVLink
- Usual allocation: treat 2 GPUs as the default; 4 GPUs are an optional fast path
- Home filesystem: about 16TB free
- Existing environment: Python 3.11.15, Torch 2.5.1+cu121, CXX11 ABI false
- Existing issue: uninstall the unrelated `hf==1.27.0` package and pin
  `huggingface-hub==0.33.4`
- Existing `/home/chunyuan/rubric` is a roughly 10GB legacy Qwen3-4B workspace;
  preserve it before cloning this repository.

## 1. Clone code without overwriting the legacy workspace

```bash
mv /home/chunyuan/rubric /home/chunyuan/rubric_legacy_qwen4b
git clone https://ghproxy.com/https://github.com/BoWang222/rubric.git /home/chunyuan/rubric
cd /home/chunyuan/rubric
```

## 2. Repair and freeze the training environment

Keep the target server's cu121 Torch build. Do not replace it with the old
server's cu124 build.

```bash
conda activate rubric
python -m pip uninstall -y hf
python -m pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements/train.txt
python -m pip install --no-deps -e .
python -m pip check
```

Transfer the already-built Torch 2.5 / Python 3.11 / ABI-false FlashAttention
wheel through the local workstation:

```bash
scp autodl_4:/root/autodl-tmp/wheels/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl /tmp/
scp /tmp/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl nus-container:/home/chunyuan/wheels/
```

Then, on `nus-container`:

```bash
conda activate rubric
python -m pip install --no-deps /home/chunyuan/wheels/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
python -c 'import torch, flash_attn; print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI, flash_attn.__version__)'
```

If that binary wheel does not import on cu121, compile only FlashAttention:

```bash
python -m pip uninstall -y flash-attn
MAX_JOBS=16 python -m pip install --no-build-isolation --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple flash-attn==2.7.4.post1
```

## 3. Create the isolated vLLM environment

```bash
conda create -n rubric_vllm python=3.11.15 pip -y
conda activate rubric_vllm
python -m pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r /home/chunyuan/rubric/requirements/vllm.txt
python -m pip check
python -c 'import torch, vllm, transformers; print(torch.__version__, vllm.__version__, transformers.__version__)'
```

This environment is for generation/judging only. Never install vLLM into the
`rubric` training environment.

## 4. Clone and pin reference repositories

```bash
cd /home/chunyuan/rubric
mkdir -p refs

git clone https://ghproxy.com/https://github.com/agentscope-ai/OpenJudge.git refs/OpenJudge
git -C refs/OpenJudge checkout --detach 2151def3553e5521ff8b3e2fea837561c57255f9

git clone https://ghproxy.com/https://github.com/Haoxiang03/RUBRIC-ARROW.git refs/RUBRIC-ARROW
git -C refs/RUBRIC-ARROW checkout --detach d116811f6e804f550f05e0f7ac5036acc7537d67

git clone https://ghproxy.com/https://github.com/eric-mitchell/direct-preference-optimization.git refs/dpo
git -C refs/dpo checkout --detach f8b8c0f49dc92a430bae41585f9d467d3618fe2f

git clone https://ghproxy.com/https://github.com/stellalisy/EvoLM.git refs/evolm
git -C refs/evolm checkout --detach 207cf7ea1e08dbca43f3396ba18733f7f7fba643

git clone https://ghproxy.com/https://github.com/kykim0/margin-matching-pref-opt.git refs/mmpo
git -C refs/mmpo checkout --detach ef3a91dfe1707ce09d4442b0a13285f87f9ae636

git clone https://ghproxy.com/https://github.com/rycolab/odpo.git refs/odpo
git -C refs/odpo checkout --detach 6152f67c8cddc223ca5affc7e261d967568068ee

git clone https://ghproxy.com/https://github.com/wanghaoyu0408/OpenRubrics.git refs/openrubrics
git -C refs/openrubrics checkout --detach 1a40c14cb7827ae10bfdf0f6d61ca35a36770b5b

git clone https://ghproxy.com/https://github.com/viswavi/RLCF.git refs/rlcf
git -C refs/rlcf checkout --detach 73254b6e4d27d769523d8e05527cb7b0eb8bcc04

git clone https://ghproxy.com/https://github.com/huggingface/trl.git refs/trl
git -C refs/trl checkout --detach accf7383a33c618a2edc7205107120b5f32e28a3
```

These repositories are read-only provenance/oracles. Do not install their old
dependency sets into either conda environment.

## 5. Download the model and datasets from the mainland mirror

```bash
cd /home/chunyuan/rubric
mkdir -p models data/raw

HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --local-dir models/qwen3-8b

HF_ENDPOINT=https://hf-mirror.com huggingface-cli download openbmb/UltraFeedback --repo-type dataset --revision 40b436560ca83a8dba36114c22ab3c66e43f6d5e --local-dir data/raw/ultrafeedback_raw
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download HuggingFaceH4/ultrafeedback_binarized --repo-type dataset --revision 3949bf5f8c17c394422ccfab0c31ea9c20bdeb85 --local-dir data/raw/ultrafeedback_binarized
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download nvidia/HelpSteer2 --repo-type dataset --revision 990b2711a36180dd19d9c94b8627844866f8982a --local-dir data/raw/helpsteer2
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download judgmentbench/JudgmentBench --repo-type dataset --revision 945ff52f4c63c17006dc30ec35fefbddf3ccf58d --local-dir data/raw/judgmentbench
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download allenai/WildChat-1M --repo-type dataset --revision 7d6490e462285cf85d91eabea0f9a954fbddcd1f --local-dir data/raw/wildchat_1m
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download viswavi/wildchecklists --repo-type dataset --revision f4175828be014a420b2729d1042c7639b5ce0e16 --local-dir data/raw/wildchecklists
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download agentscope-ai/Auto-Rubric --repo-type dataset --revision 2a5e18a3a1d57367bb907cf7a40ce33219e29385 --local-dir data/raw/auto_rubric
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download OpenRubrics/OpenRubric-v2 --repo-type dataset --revision d1048e9e0375034d1dafb53392789cd368424861 --local-dir data/raw/openrubric_v2
```

Expected download sizes from the source machine are approximately 16GB model,
5GB raw datasets, 400MB reference repositories, and 147MB processed
UltraFeedback data.

## 6. Rebuild deterministic UltraFeedback products

Rebuild the tokenized data from pinned raw inputs. Rebuild reference logps on
H100/cu121 rather than silently reusing the A800/cu124 cache.

```bash
conda activate rubric
cd /home/chunyuan/rubric

rubric-prepare-ultrafeedback \
  --raw-dir data/raw/ultrafeedback_raw \
  --h4-dir data/raw/ultrafeedback_binarized \
  --model models/qwen3-8b \
  --output data/processed/ultrafeedback/v2

rubric-verify-source-parity \
  --project-root /home/chunyuan/rubric \
  --output artifacts/baselines/qwen3_8b_ultrafeedback_margin_consumers_v1/source_parity.json

torchrun --nproc_per_node=2 -m rubric_dpo.cli.cache_reference \
  --dataset-dir data/processed/ultrafeedback/v2/tokens_qwen3_8b_non_thinking \
  --model models/qwen3-8b \
  --output data/cache/ultrafeedback_qwen3_8b_ref_v1

rubric-verify-reference-cache \
  --model models/qwen3-8b \
  --dataset-dir data/processed/ultrafeedback/v2/tokens_qwen3_8b_non_thinking \
  --reference-cache data/cache/ultrafeedback_qwen3_8b_ref_v1 \
  --output data/cache/ultrafeedback_qwen3_8b_ref_v1/verification.json

rubric-preflight --root /home/chunyuan/rubric --min-gpus 2
```

## 7. GPU launch profiles

Two-GPU H100 is the conservative default and requires CPU parameter offload.
It must pass smoke before pilots or full runs:

```bash
rubric-launch smoke \
  --root /home/chunyuan/rubric \
  --gpu-count 2 \
  --gpu-ids 0,1 \
  --two-gpu-offload \
  --parallel off
```

When four GPUs are allocated, use the faster no-offload profile:

```bash
rubric-launch smoke \
  --root /home/chunyuan/rubric \
  --gpu-count 4 \
  --gpu-ids 0,1,2,3 \
  --parallel off
```

Do not run two independent one-GPU full-parameter jobs. If the two-GPU offload
smoke fails, wait for a four-GPU allocation rather than changing the research
batch or mixing LoRA with full-parameter baselines.
