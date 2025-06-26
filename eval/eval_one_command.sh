#!/bin/bash

# Default values
model="gaotang/RM-R1-Qwen2.5-Instruct-32B"
model_save_name="RM-R1-Qwen2.5-Instruct-32B"
device="0,1,2,3,4,5,6,7"
vllm_gpu_util=0.90
num_gpus=8
max_tokens=50000

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) model="$2"; shift ;;
        --model_save_name) model_save_name="$2"; shift ;;
        --device) device="$2"; shift ;;
        --vllm_gpu_util) vllm_gpu_util="$2"; shift ;;
        --num_gpus) num_gpus="$2"; shift ;;
        --max_tokens) max_tokens="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Print the arguments
echo "Model: $model"
echo "Model Save Name: $model_save_name"
echo "Device: $device"
echo "VLLM GPU Util: $vllm_gpu_util"
echo "Num GPUs: $num_gpus"
echo "Max Tokens: $max_tokens"

# Record the absolute path of the current directory (RM-R1)
CUR_DIR="$(pwd)"
META_RESULT_SAVE_DIR="${CUR_DIR}/eval/result"


##################################
########## Reward Bench ##########
##################################

cd ${CUR_DIR}/eval/reward-bench
echo $PWD

CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py \
    --model $model \
    --vllm_gpu_util $vllm_gpu_util \
    --trust_remote_code \
    --model_save_name $model_save_name \
    --meta_result_save_dir $META_RESULT_SAVE_DIR \
    --num_gpus=$num_gpus \
    --max_tokens=$max_tokens \

##################################
############ RM-Bench ############
################################## 

cd ${CUR_DIR}/eval/RM-Bench
echo $PWD

CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py --trust_remote_code --model_save_name $model_save_name --model $model --datapath data/total_dataset_1.json --vllm_gpu_util $vllm_gpu_util --num_gpus=$num_gpus --max_tokens=$max_tokens --META_RESULT_SAVE_DIR $META_RESULT_SAVE_DIR
CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py --trust_remote_code --model_save_name $model_save_name --model $model --datapath data/total_dataset_2.json --vllm_gpu_util $vllm_gpu_util --num_gpus=$num_gpus --max_tokens=$max_tokens --META_RESULT_SAVE_DIR $META_RESULT_SAVE_DIR
CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py --trust_remote_code --model_save_name $model_save_name --model $model --datapath data/total_dataset_3.json --vllm_gpu_util $vllm_gpu_util --num_gpus=$num_gpus --max_tokens=$max_tokens --META_RESULT_SAVE_DIR $META_RESULT_SAVE_DIR

python scripts/process_final_result.py --model_save_name $model_save_name --model $model --meta_result_save_dir $META_RESULT_SAVE_DIR

##################################
############### RMB ##############
################################## 

cd ${CUR_DIR}/eval/RMB-Reward-Model-Benchmark
echo $PWD

CUDA_VISIBLE_DEVICES=$device python eval/scripts/run_generative.py \
    --model $model \
    --num_gpus=$num_gpus \
    --trust_remote_code \
    --model_save_name $model_save_name \
    --vllm_gpu_util $vllm_gpu_util \
    --meta_result_save_dir $META_RESULT_SAVE_DIR \
    --dataset_dir RMB_dataset/Pairwise_set/Harmlessness \
    --max_tokens=$max_tokens 

CUDA_VISIBLE_DEVICES=$device python eval/scripts/run_generative.py \
    --model $model \
    --num_gpus=$num_gpus \
    --trust_remote_code \
    --model_save_name $model_save_name \
    --vllm_gpu_util $vllm_gpu_util \
    --meta_result_save_dir $META_RESULT_SAVE_DIR \
    --max_tokens=$max_tokens 

CUDA_VISIBLE_DEVICES=$device python eval/scripts/run_generative_bestofn.py \
    --model $model \
    --num_gpus=$num_gpus \
    --trust_remote_code \
    --model_save_name $model_save_name \
    --vllm_gpu_util $vllm_gpu_util \
    --meta_result_save_dir $META_RESULT_SAVE_DIR \
    --dataset RMB_dataset/BoN_set/Helpfulness \
    --max_tokens=$max_tokens 

CUDA_VISIBLE_DEVICES=$device python eval/scripts/run_generative_bestofn.py \
    --model $model \
    --num_gpus=$num_gpus \
    --trust_remote_code \
    --model_save_name $model_save_name \
    --vllm_gpu_util $vllm_gpu_util \
    --meta_result_save_dir $META_RESULT_SAVE_DIR \
    --dataset RMB_dataset/BoN_set/Harmlessness \
    --max_tokens=$max_tokens 

python eval/scripts/process_final_result.py --model_save_name $model_save_name --meta_result_save_dir $META_RESULT_SAVE_DIR