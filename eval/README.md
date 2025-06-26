# Evaluations for Reward Models

This directory provides the one-command evaluation code for Reward Models. It is adapted from Reward-Bench, RM-Bench, and RMB. We sincerely thank the authors of these benchmarks for their contributions. Compared with the standard score-only evaluation pipeline, we log per-sample output of the evaluation result to promote better transparency and evaluation of generative reward models. The examplar cases are presented in [result](/result). We generally adapt the original codebase without too much changes, except for RMB, where we re-implement their whole pipelines. A more detailed explanation is provided in xxx.

### Requirements 

The code has been tested with the following main dependencies (there is no need to `pip install -e .`):

```bash
torch=2.6.0cu12.4+
transformers=4.51.0
vllm=0.8.3
fastchat
accelerate==1.8.1
datasets
openai
google-generativeai
anthropic
together
```

The main conflict to be aware of is that the `transformer` and `vllm` might be imcompitble with each other.


### Running the evaluation

Our evaluation supports one-command running. An example is provided below:

```bash
bash eval_one_command.sh --model gaotang/RM-R1-DeepSeek-Distilled-Qwen-32B --model_save_name RM-R1-Deepseek-Distilled-32B --device 0,1,2,3 --vllm_gpu_util 0.90 --num_gpus 4
```

### Community Contributing 

We believe in the power of open-evaluation and that the comprehensive benchmarks used in this paper can be applied to any existing research in reward modeling. Feel free to submit pull requests to **add your models to our codebase** or **add new benchmarks** with our one-command multi-benchmark evaluation. 