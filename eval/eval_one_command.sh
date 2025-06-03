device="0,1,2,3,4,5,6,7"
model="gaotang/RM-R1-Qwen2.5-Instruct-32B"
vllm_gpu_util=0.85
model_save_name="RM-R1-Qwen2.5-Instruct-32B"
num_gpus=8
max_tokens=50000

# Record the absolute path of the current directory (RM-R1)
CUR_DIR="$(pwd)"
META_RESULT_SAVE_DIR="${CUR_DIR}/eval/result"


##################################
########## Reward Bench ##########
##################################

cd eval/reward-bench
echo $PWD

CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py \
    --model $model \
    --vllm_gpu_util $vllm_gpu_util \
    --trust_remote_code \
    --model_save_name $model_save_name \
    --meta_result_save_dir $META_RESULT_SAVE_DIR \
    --num_gpus=$num_gpus \
    --max_tokens=$max_tokens \