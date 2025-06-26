# RM-Bench

This repository contains the data of the ICLR 24 Oral Paper "*RM-Bench: Benchmarking Reward Models of Language Models with Subtlety and Style*". RM-Bench is a benchmark dataset for evaluating reward models of language modeling. It focuses on two aspects of reward models: **Sensitivity to Subtle Changes** and **Robustness to Style Bias**.
Specifically, for each prompt in RM-Bench, it provides three chosen responses and three rejected responses with different styles.
The difference between the chosen and rejected responses is subtle, and the styles of the responses are varied from concise to detailed to well-formatted.


<img src="https://github.com/THU-KEG/RMBench/blob/main/assets/example_data.png?raw=true" alt="Example Data" width="800"/>
<p style="text-align: center;"><em>Figure 1: Example Data from RMBench. The rejected response incorrect because Schrödinger's cat illustrates the concept of quantum superposition, not quantum entanglement.
$y^\varnothing$ is a concise response, $y^{\text{L}}$ is a detailed response, and $y^{\text{L,M}}$ is a detailed response with markdown formatting.
</em></p>

A more detailed explanation of the code structure can be found in the original [RM-Bench codebase](https://github.com/THU-KEG/RM-Bench), and we sincerely appreciate their contributions. 

In addition to the scalar-based evaluation pipeline provided by RM-Bench, this repository extends the benchmark by implementing a complete evaluation pipeline for generative reward models. The generative evaluation can be run using the script [`eval_one_command.sh`](../eval_one_command.sh).