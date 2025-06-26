#!/bin/bash

# Default values
model="Skywork/Skywork-Reward-Llama-3.1-8B"
model_save_name="Skywork/Skywork-Reward-Llama-3.1-8B"
device="0,1,2,3"

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) model="$2"; shift ;;
        --model_save_name) model_save_name="$2"; shift ;;
        --device) device="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Print the arguments
echo "Model: $model"
echo "Model Save Name: $model_save_name"
echo "Device: $device"

# Record the absolute path of the current directory (RM-R1)
CUR_DIR="$(pwd)"
META_RESULT_SAVE_DIR="${CUR_DIR}/eval/result"

echo "Meta Result Save Directory: $META_RESULT_SAVE_DIR"

##################################
############### RMB ##############
##################################

cd ${CUR_DIR}/eval/RMB-Reward-Model-Benchmark
echo $PWD 

CUDA_VISIBLE_DEVICES=$device python eval/scripts/run_rm.py \
    --model $model \
    --trust_remote_code \
    --model_save_name $model_save_name \
    --meta_result_save_dir $META_RESULT_SAVE_DIR \
    --dataset_dir RMB_dataset/Pairwise_set/Harmlessness \
    --batch_size 8

CUDA_VISIBLE_DEVICES=$device python eval/scripts/run_rm.py \
    --model $model \
    --trust_remote_code \
    --model_save_name $model_save_name \
    --meta_result_save_dir $META_RESULT_SAVE_DIR \
    --dataset_dir RMB_dataset/Pairwise_set/Helpfulness \
    --batch_size 8

CUDA_VISIBLE_DEVICES=$device python eval/scripts/run_rm_bestofn.py \
    --model $model \
    --trust_remote_code \
    --model_save_name $model_save_name \
    --meta_result_save_dir $META_RESULT_SAVE_DIR \
    --dataset RMB_dataset/BoN_set/Harmlessness \
    --batch_size 8

CUDA_VISIBLE_DEVICES=$device python eval/scripts/run_rm_bestofn.py \
    --model $model \
    --trust_remote_code \
    --model_save_name $model_save_name \
    --meta_result_save_dir $META_RESULT_SAVE_DIR \
    --dataset RMB_dataset/BoN_set/Helpfulness \
    --batch_size 8


python eval/scripts/process_final_result.py --model_save_name $model_save_name --meta_result_save_dir $META_RESULT_SAVE_DIR
