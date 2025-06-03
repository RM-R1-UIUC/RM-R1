```
CUDA_VISIBLE_DEVICES=1 python /shared/nas2/xiusic/Rubric-RM/evaluation/reward-bench/scripts/run_generative.py --model wzq016/llama3-skywork-rlrm-filtered-grpo-kl --stop_llamma3 --model_save_name filtered_llama3 --rubric_rl_new --vllm_gpu_util 0.75

CUDA_VISIBLE_DEVICES=7 python /shared/nas2/xiusic/Rubric-RM/evaluation/reward-bench/scripts/run_generative.py --model wzq016/llama3-skywork-rlrm-filtered-code-grpo-kl --stop_llamma3 --model_save_name filtered_llama3_code --rubric_rl_new

CUDA_VISIBLE_DEVICES=6 python /shared/nas2/xiusic/Rubric-RM/evaluation/reward-bench/scripts/run_generative.py --model /shared/nas2/xiusic/gaotang/skylab-v02-math-18k-code-2_5k-baseline/ckpt/qwen-14b-sft --model_save_name filtered_qwen_14b_code_math-sft --sft_new --vllm_gpu_util 0.9

CUDA_VISIBLE_DEVICES=7 python /shared/nas2/xiusic/Rubric-RM/evaluation/reward-bench/scripts/run_generative.py --model wzq016/qwen25-code-math-evidence-rubric --model_save_name sky_math_code_qwen7b_original_length1024 --rubric_evidence

CUDA_VISIBLE_DEVICES=3 python /shared/nas2/xiusic/Rubric-RM/evaluation/reward-bench/scripts/run_generative.py --model /shared/nas2/xiusic/gaotang/skylab-v02-math-18k-code-2_5k-baseline/ckpt/qwen-7b-sft-adjusted --sft_new --model_save_name filtered_qwen_7b_code_math-sft


CUDA_VISIBLE_DEVICES=7 python /shared/nas2/xiusic/Rubric-RM/evaluation/reward-bench/scripts/run_generative.py --model wzq016/qwen25-code-math-evidence-rubric-4k2k --model_save_name sky_math_code_qwen7b_original_4k2k --rubric_evidence --vllm_gpu_util 0.45

CUDA_VISIBLE_DEVICES=7 python /shared/nas2/xiusic/Rubric-RM/evaluation/reward-bench/scripts/run_generative.py --model wzq016/qwen25-code-math-evidence-rubric-4k2k-separate --model_save_name sky_math_code_qwen7b_original_4k2k_separate --rubric_evidence --vllm_gpu_util 0.45

CUDA_VISIBLE_DEVICES=1 python /shared/nas2/xiusic/Rubric-RM/evaluation/reward-bench/scripts/run_generative.py --model /shared/nas2/xiusic/gaotang/ckpt/other_run/qwen2.5_7B_LR5.0e-6_evidence_rubric_4k4k_separate_reward_function_largeBz/global_step_36 --model_save_name sky_math_code_qwen7b_original_4k4k_separate_largeBz --rubric_evidence --vllm_gpu_util 0.7

CUDA_VISIBLE_DEVICES=2 python /shared/nas2/xiusic/Rubric-RM/evaluation/reward-bench/scripts/run_generative.py --model /shared/nas2/xiusic/gaotang/skylab-v02-math-18k-code-2_5k-baseline/qwen2.5_14B_LR1.0e-6_evidence_rubric_4k2k_separate_reward_function/global_step_73/actor/huggingface --model_save_name 14b_4k2k_evidence_separate_reward --rubric_evidence --vllm_gpu_util 0.9 --trust_remote_code
```