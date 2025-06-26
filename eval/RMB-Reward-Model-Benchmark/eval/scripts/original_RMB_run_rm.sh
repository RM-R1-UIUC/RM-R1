# your models root dir
# If you want download model from huggingface directly, no need to fill model_path
model_path=''
models=(
    'ArmoRM-Llama3-8B-v0.1'
    # 'Eurus-RM-7b'
    # 'stablelm-2-12b-chat'
    # 'Starling-RM-34B'
    # 'internlm2-7b-reward'
    # 'internlm2-20b-reward'
    # 'tulu-v2.5-13b-preference-mix-rm'
)

# your RMB_dataset path
dataset_path=''
datasets=(
    'Pairwise_set/Helpfulness/Brainstorming/Idea Development.json'
)

# your results path
result_path='../RMB-Reward-Model-Benchmark/eval/results'

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "$dataset"
        echo "$model"
        python RMB-Reward-Model-Benchmark/eval/scripts/my_run_rm.py \
        --model_dir="${model_path}" \
        --model="${model_path}/${model}" \
        --dataset_dir="${dataset_path}/${dataset}" \
        --single_data=True \
        --dataset="${dataset_path}/${dataset}/${dataset}.json" \
        --results="${result_path}/${model}_result.json"
    done
done


# > "log/eval".log 2>&1 &