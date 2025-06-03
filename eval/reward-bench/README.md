This code implements the evaluation pipeline of `reward_bench`. A typical typical prompt looks like the following 

```
# SFT:
CUDA_VISIBLE_DEVICES=1 python scripts/run_generative.py --model $model_path --model_save_name $save_name --sft_new --vllm_gpu_util 0.9

# RL:
CUDA_VISIBLE_DEVICES=1 python scripts/run_generative.py --model $model_path --model_save_name $save_name --rubric_rl_new --vllm_gpu_util 0.9
```

Examples of current sft:

```
CUDA_VISIBLE_DEVICES=0 python scripts/run_generative.py --model /shared/nas2/xiusic/gaotang/skylab-v02-math-18k-code-2_5k-baseline/ckpt/llama3-8b-sft-adjusted --model_save_name llama3-8b-sft-math-code-sky --sft_new --vllm_gpu_util 0.9 --stop_llamma3

CUDA_VISIBLE_DEVICES=0 python scripts/run_generative.py --model /shared/nas2/xiusic/gaotang/skylab-v02-math-18k-code-2_5k-baseline/ckpt/qwen-7b-sft-adjusted --model_save_name qwen-7b-sft-math-code-sky --sft_new --vllm_gpu_util 0.9

CUDA_VISIBLE_DEVICES=0 python scripts/run_generative.py --model /shared/nas2/xiusic/gaotang/skylab-v02-math-18k-code-2_5k-baseline/ckpt/qwen-14b-sft-adjusted --model_save_name qwen-14b-sft-math-code-sky --sft_new --vllm_gpu_util 0.9
```

ICL vs original:

```
CUDA_VISIBLE_DEVICES=0 python scripts/run_generative.py --model Qwen/Qwen2.5-7B-Instruct --model_save_name qwen-7b-original --sft_new --vllm_gpu_util 0.45

CUDA_VISIBLE_DEVICES=5 python scripts/run_generative.py --model Qwen/Qwen2.5-7B-Instruct --model_save_name qwen-7b-original --icl --vllm_gpu_util 0.85

CUDA_VISIBLE_DEVICES=5 python scripts/run_generative.py --model /shared/nas2/xiusic/gaotang/skylab-v02-math-18k-code-2_5k-baseline/ckpt/qwen-7b-sft-adjusted --model_save_name qwen-7b-sft --icl --vllm_gpu_util 0.85
```

Guideline: 

```
CUDA_VISIBLE_DEVICES=6 python scripts/run_generative.py --model wzq016/qwen25-entrie-guideline-8k --model_save_name entire_guideline_76k --guideline

CUDA_VISIBLE_DEVICES=7 python scripts/run_generative.py --model wzq016/qwen25-filtered-guideline-8k --model_save_name filtered_guideline_55k --guideline
```