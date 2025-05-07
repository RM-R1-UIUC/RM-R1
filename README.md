# RM-R1: **Reward Modeling as Reasoning**

[**🤗 Model**](https://huggingface.co/collections/gaotang/rm-r1-681128cdab932701cad844c8) | [**📊 Dataset**](https://huggingface.co/collections/gaotang/rm-r1-681128cdab932701cad844c8) | [**📖 Paper**](https://arxiv.org/abs/2505.02387)

<p align="center">
  <img src="figures/rm-r1-1.png" alt="RM‑R1 pipeline" width="80%"/>
</p>


**RM‑R1** reframes reward modeling as a *reasoning* problem. Instead of emitting an opaque scalar, a Reasoning Reward Model (ReasRM) first *thinks out loud*—generating a structured rubric or solution—and then predicts the preference between two responses. This simple shift boosts both *interpretability* **and** *performance*: RM‑R1 beats prior open‑source reward models (e.g. GPT-4o, Llama3.1-405B) on multiple public benchmarks, while letting you read *why* the model prefers one answer over the other.  

This repository provides all materials necessary to reproduce and extend RM-R1:

- End-to-end scripts and configs for training (Distillation + RL),
- a unified evaluation harness for public benchmarks, and 
- ready-to-run examples for deployment and inference.

All experiments are fully documented so that results can be audited or adapted to new domains with minimal changes. 

**<span style="color:#d45500;">⚡ This repository is continuously updated—star it to follow new releases!</span>**

---

## 📑 Table of Contents
1. [Installation](#installation)
2. [Training](#-training-workflow)
3. [Evaluation](#evaluation)
4. [User Our Model](#use-our-model)
5. [Build Your Own Dataset](#-build-your-own-dataset)
6. [Features](#features)
7. [Acknowledgements](#acknowledgement)
8. [Citation](#citations)

---


## Installation

> **Important**: RM‑R1 currently depends on **specific commits** of veRL and vLLM. Please follow the exact steps below—even if you already have vLLM installed—otherwise compilation or runtime errors may occur.

### 1. Base environment
```bash
# create and enter env (Python ≥3.11 recommended)
conda create -n rm-r1 python=3.11 -y
conda activate rm-r1
```

### 2. veRL – pinned commit
```bash
git clone https://github.com/volcengine/verl
cd verl
git checkout e49fb572bf85a8f0ef7124c898f509bd6d9832a1
pip install -e .
cd ..
```

### 3. vLLM – pinned commit + flash‑attention
```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout ed6e9075d31e32c8548b480a47d1ffb77da1f54c
git cherry-pick caac5c2e597b1780c3df54a537c34e6061c32cff
export VLLM_COMMIT=ed6e9075d31e32c8548b480a47d1ffb77da1f54c
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/ed6e9075d31e32c8548b480a47d1ffb77da1f54c/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
VLLM_USE_PRECOMPILED=1 pip install --editable .

# flash‑attention 2 (for >2× speed‑up)
pip install flash-attn==2.7.2.post1 --no-build-isolation
```

**Done!** You can now run RM‑R1 for RL training.

### (Optional) Distillation / SFT environment

If you intend to reproduce the *reasoning‑distillation* stage from scratch, we recommend a separate environment:

```bash
conda create -n rm-r1-sft python=3.11 -y
conda activate rm-r1-sft

pip install uv && uv pip install --upgrade pip
uv pip install vllm==0.7.2

# OpenRLHF
cd OpenRLHF
uv pip install -e .
```

---

## 🚀 Training Workflow

All training recipes live in [`rm_r1/scripts/`](rm_r1/scripts/). The pipeline has **two stages** (for instruct models):

| Stage | Script Directory | 
|-------|----------------|
| **Distillation** | `scripts/Distill/distill_qwen2.5-*.sh` |
| **RL with Verifiable Rewards (RLVR)** | `scripts/RLVR/*train_rm_r1_rlvr_*.sh` | 

Specify `SAVE_MODEL_PATH` in every distillation script and `SAVE_META_DIR` in every RLVR script to choose where checkpoints are stored. Other arguments such as batch size, learning rate, Slurm partition, etc., can be edited directly in each shell script. Detailed flag descriptions are available in the [veRL documentation](https://verl.readthedocs.io/en/latest/index.html).

### 🔧 Example  Training a 14 B *Instruct* model from scratch

```bash
# 1️⃣ Distillation (SFT)
conda activate rm-r1-sft
cd rm_r1/OpenRLHF
bash ../scripts/Distill/local/distill_qwen2.5-14b-instruct.sh

# 2️⃣ RLVR fine‑tuning
conda deactivate
conda activate rm-r1
cd ../..

# – local
bash rm_r1/scripts/RLVR/local/train_rm_r1_rlvr_qwen2.5_instruct_14b.sh

# – Slurm cluster
sbatch rm_r1/scripts/RLVR/slurm/train_rm_r1_rlvr_qwen2.5_instruct_14b.sh
```

### 🔧 Example  Fine‑tuning a **DeepSeek‑distilled** checkpoint

```bash
conda activate rm-r1

# – local
bash rm_r1/scripts/RLVR/local/train_rm_r1_rlvr_dpsk_distilled_14b.sh

# – Slurm
sbatch rm_r1/scripts/RLVR/slurm/train_rm_r1_rlvr_dpsk_distilled_14b.sh
```

---


## Use Our Model 

- coming soon

---

## Evaluation 

- coming soon

---

## 🧩 Build Your Own Dataset

This section outlines how we curated and blended data for training **RM‑R1**, and how you can adapt the process for your own use case.


### 🔗 Source Pools

We mix examples from the following datasets:

| Dataset | Size | Domain |
|--------|------|--------|
| **[Skywork‑Reward‑Preference‑80K‑v0.2](https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2)** | ~54k pairs | General |
| **[Code‑Preference‑Pairs](https://huggingface.co/datasets/Vezora/Code-Preference-Pairs)** | 8k pairs | Code |
| **[Math‑Step‑DPO‑10K](https://huggingface.co/datasets/xinlai/Math-Step-DPO-10K)** | 10k pairs | Math |

We thank the authors of these datasets for their contributions. If you'd like to construct your own dataset, feel free to refer to our mixing script at [`rm_r1/dataset/mix_data/mix_data.py`](rm_r1/dataset/mix_data/mix_data.py).

### 🛠️ Generating High‑Quality Reasoning Chains

A key component of RM‑R1 training is the use of **correct and coherent** distilled reasoning chains. Naively prompting strong models (e.g., O3, Claude) in a zero-shot setting yields only ~75% chain accuracy—insufficient for stable RL training. To address this, we use a **two-pass bootstrapping strategy**:

1. **Pass 1 (Claude-3.7-Sonnet):** Generate chains via zero-shot prompting.
2. **Keep:** Retain samples with incorrect answers and their corresponding chains.
3. **Pass 2 (O3):** Provide the *correct* answer, the prompt, and the flawed chain from Pass 1. Ask the model to regenerate a corrected reasoning chain.

This approach reliably produces chains that are both accurate and logically sound. Our implementations are provided at [`rm_r1/dataset/reasoning_chain_generation/`](rm_r1/dataset/reasoning_chain_generation/).


---


## Features 

- Open release of trained model and the full accompanying datasets. ✔️ 
- End-to-end pipelines for both supervised fine-tuning (SFT) and reinforcement learning (RL). ✔️ 
- Support different RL frameworks. ✔️   
- Support Slurm v.s. Interactive Training. ✔️ 
- Support multi-node, multi-gpu training. ✔️ 
- Support different LLMs. ✔️ 
- Support building your own custom dataset. 
- One-command evaluation on public RM benchmarks for quick, reproducible reporting.

---

## Acknowledgement 

The concept of RM-R1 is inspired by [Deepseek-R1](https://github.com/deepseek-ai/DeepSeek-R1). Its implementation is built upon [veRL](https://github.com/volcengine/verl) and [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). We sincerely appreciate the efforts of these teams for their contributions to open-source research and development.

---

## Citations

```bibtex
@article{chen2025rm,
  title={RM-R1: Reward Modeling as Reasoning},
  author={Chen, Xiusi and Li, Gaotang and Wang, Ziqi and Jin, Bowen and Qian, Cheng and Wang, Yu and Wang, Hongru and Zhang, Yu and Zhang, Denghui and Zhang, Tong and others},
  journal={arXiv preprint arXiv:2505.02387},
  year={2025}
}
```
