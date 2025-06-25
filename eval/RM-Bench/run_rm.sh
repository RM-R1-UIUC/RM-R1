#!/bin/bash

# Prompt for model path
read -p "Enter the model path (default: RewardModels/allenai/tulu-v2.5-13b-hh-rlhf-60k-rm): " MODEL_PATH
MODEL_PATH=${MODEL_PATH:-RewardModels/allenai/tulu-v2.5-13b-hh-rlhf-60k-rm}

# Prompt for CUDA device
read -p "Enter the CUDA device (default: 0): " CUDA_DEVICE
CUDA_DEVICE=${CUDA_DEVICE:-0}

# source /mnt/wuxuaner/workspace/miniconda3/bin/activate torch231
# source activate torch231
# export PYTHONPATH=$PYTHONPATH:pwd
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

chat_template=tulu
python -m scripts/run_rm \
    --model $MODEL_PATH \
    --datapath data/total_dataset.json \
    --batch_size 8 \
    --trust_remote_code \
    --chat_template $chat_template \