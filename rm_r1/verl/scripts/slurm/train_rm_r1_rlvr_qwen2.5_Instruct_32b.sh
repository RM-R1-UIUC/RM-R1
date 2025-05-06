#!/bin/bash
#SBATCH --job-name=rl                                # Job name
#SBATCH --nodes=4                                    # Number of nodes
#SBATCH --ntasks-per-node=1                          # Number of tasks per node
#SBATCH --cpus-per-task=128                          # Number of CPUs per task
#SBATCH --gres=gpu:8                                 # Number of GPUs per node
#SBATCH --gpus-per-task=8                            # Number of GPUs per task
#SBATCH --exclusive                                  # Use the entire node, all CPUs and memory
#SBATCH --time=240:00:00                              # Maximum runtime
#SBATCH --output=./logs/output_%j.log                # Standard output and error log

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export VERL_PPO_LOGGING_LEVEL="INFO"
export PYTHONUNBUFFERED=1
N_GPU=8
N_NODES=4


# ========SETTING RAY CLUSTER============
# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Killing previous ray cluster at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray stop
sleep 5

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus 128 --num-gpus "${N_GPU}" --block &


sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Killing previous ray cluster at Worker $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray stop
    sleep 5

    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus 128 --num-gpus "${N_GPU}" --block &
    sleep 5
done



# ============SETTING PARAMETER===============
# Model Setting
MODEL_PATH=your_distilled_model_path # Distilled Model Path 

# Training Setting
LR=5.0e-7
GPU_MEM_UTILIZATION=0.4 # Lower this if you met OOM problem
TOTAL_EPISODES=1
SAVE_EVERY_STEP=45
TEST_EVERY_STEP=100000
TRAIN_BS=1024           # Rollout batchsize. Could be arbitrary large, but must be divided by N_GPU.
PPO_MINI_BS=128         # Train batch size. Could be arbitrary large, must be the divisor of TRAIN_BS and be divided by N_GPU. Setting this equal to TRAIN_BS means strictly on-policy.

MAX_PROMPT_LENGTH=4096  # Lower this if you met OOM problem.
MAX_RESPONSE_LENGTH=8192 # Lower this if you met OOM problem
TRAIN_PER_GPU=1         # REAL train batch size per gpu. Lower this if you met OOM problem. Must be a divisor of PPO_MINI_BS.
FORWARD_PER_GPU=1       # Batch size to get logprob. Lower this if you met OOM problem. Must be a divisor of TRAIN_BS.

# Logging Setting
PROJECT_NAME=RM-R1
EXPERIMENT_NAME=qwen2.5_32B_LR${LR}_32B_RLVR

# Reward Setting
REWARD_PATH=./rm_r1/verl/utils/reward_score/lm_as_judge.py
REWARD_FUNC_NAME=lm_as_judge_match

# Task
TRAIN_TASK="#Your Data"
EVAL_TASK="#Your Data"

# FIXED SETTING (DO NOT MODIFY IF YOU DO NOT KNOW WHAT IT MEANS)
MAX_NUM_BATCHED_TOKENS=$(($MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH))


python3 -m rm_r1.verl.trainer.main_ppo \
    data.train_files=${TRAIN_TASK} \
    data.val_files=${EVAL_TASK} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.train_batch_size=${TRAIN_BS} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BS} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${TRAIN_PER_GPU} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTILIZATION} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${FORWARD_PER_GPU} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${FORWARD_PER_GPU} \
    custom_reward_function.path=${REWARD_PATH} \
    custom_reward_function.name=${REWARD_FUNC_NAME} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.total_epochs=${TOTAL_EPISODES} \
    trainer.save_freq=${SAVE_EVERY_STEP} \
    trainer.test_freq=-1 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.nnodes=${N_NODES} \
    actor_rollout_ref.actor.entropy_coeff=0