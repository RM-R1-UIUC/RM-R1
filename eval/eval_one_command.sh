model="gaotang/RM-R1-Qwen2.5-Instruct-32B"
model_save_name="RM-R1-Qwen2.5-Instruct-32B"

device="0,1,2,3,4,5,6,7"
vllm_gpu_util=0.90
num_gpus=8
max_tokens=50000

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

