# Re-Implementation of "RMB: Comprehensively Benchmarking Reward Models in LLM Alignment"

*RMB* is a comprehensive RM benchmark that covers over 49 real-world scenarios and includes both *pairwise* and *Best-of-N (BoN)* evaluations to better reflect the effectiveness of RMs in *guiding alignment optimization*. 

This directory provides a **clean re-implementation** of the RMB benchmark with the following features:

- 🧱 **Generative Evaluation Pipeline**: Written from scratch for clarity and extensibility.
- 📊 **Scalar RM Evaluation Re-Implementation**: We re-implemented the scalar-model evaluation pipeline via adapting the codebase from [RM-Bench](https://github.com/THU-KEG/RM-Bench) with our customizations.
- 📂 **Per-sample Logging**: We document detailed per-sample output for clear interpretation and better reproducibility.

The original RMB codebase can be found [here](https://github.com/Zhou-Zoey/RMB-Reward-Model-Benchmark).

## 🛠️ Running Model Evaluations 

To replicate the original setup:

- RMB’s original scalar model script is located at:  
  [`eval/scripts/original_RMB_run_rm.sh`](eval/scripts/original_RMB_run_rm.sh)

The script for evaluating generative reward models is provided in [`eval_one_command.sh`](../eval_one_command.sh).

In addition, we provide scripts for running scalar RM evaluations in [`eval/scripts/example_scalar_model_script/skywork_RMB.sh`](eval/scripts/example_scalar_model_script/skywork_RMB.sh). This script follows the same logic as our implemented generative evaluation pipeline for easy integration and experimentation.