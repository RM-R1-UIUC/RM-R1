# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# run a generative RM. For now, this requires openai and anthropic to be installed
# Examples:
# python scripts/run_generative.py --model gpt-3.5-turbo
# python scripts/run_generative.py --model=claude-3-haiku-20240307

# note: for none API models, this script uses vllm
# pip install vllm

import argparse
import logging
import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from fastchat.conversation import get_conv_template
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from my_utils.utils import torch_dtype_mapping, load_BoN_dataset

# from rewardbench import load_eval_dataset, save_to_hub
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.generative import (
    ANTHROPIC_MODEL_LIST,
    API_MODEL_LIST,
    GEMINI_MODEL_LIST,
    OPENAI_MODEL_LIST,
    format_judge_answers,
    process_judgement,
    run_judge_pair,
)
from rewardbench.utils import calculate_scores_per_section
import openai 
# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)

def find_json_files(directory):
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    return json_files

def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to model")
    parser.add_argument("--dataset_dir", type=str, default="RMB_dataset/BoN_set/Helpfulness", help="path to data_dir",
                        choices=["RMB_dataset/BoN_set/Helpfulness", "RMB_dataset/BoN_set/Harmlessness"])
    parser.add_argument("--results_meta_dir", type=str, default="eval/result", help="path to results")
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for inference")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length of RM inputs (passed to pipeline)")
    parser.add_argument(
        "--debug", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--model_save_name", type=str, default=""
    )
    parser.add_argument(
        '--max_tokens', type=int, default=2048, help='max tokens for generation'
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus to use, for multi-node vllm")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.9, help="gpu utilization for vllm")
    parser.add_argument(
        "--meta_result_save_dir", default=None, type=str
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.dataset_dir == "RMB_dataset/BoN_set/Helpfulness":
        args.model_save_meta = "BoN_set_Helpfulness"
    else:
        args.model_save_meta = "BoN_set_Harmlessness"
    ###############
    # Setup logging
    ###############
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)

    logger.info(f"Running reward model on {args.model}")

    if args.num_gpus > 1:
        # Set the environment variable
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    model = LLM(
        args.model,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.num_gpus,
        gpu_memory_utilization=args.vllm_gpu_util,
        # max_seq_length=args.vllm_max_seq_length,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    dataset_meta_dir = args.dataset_dir 
    data_path_list = find_json_files(dataset_meta_dir)


    dataset, subsets = load_BoN_dataset(
        core_set=False,
        EXTRA_PREF_SETS = data_path_list,
        conv=None,
        custom_dialogue_formatting=None,
        tokenizer=None,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "pair_uid", "category_path"],
    )

    if args.debug:
        dataset = dataset.select(range(10))
        subsets = subsets[:10]

    if "prometheus" in args.model:
        model_modifier = "prometheus"
    elif "Con-J" in args.model:
        model_modifier = "Con-J"
    elif "OffsetBias" in args.model:
        model_modifier = "offsetbias"
    elif "Atla" in args.model:
        logger.info("Using ATLA model")
        model_modifier = "Atla"
    elif "gemini" in args.model:
        model_modifier = "gemini"
    elif "RM-R1" in args.model and "Instruct" in args.model:
        model_modifier = "RM-R1-Instruct"
    elif "RM-R1" in args.model and "DeepSeek-Distilled" in args.model:
        model_modifier = "RM-R1-Reasoning"
    else:
        model_modifier = None

    ############################
    # Run model weights with vllm
    ############################

    def convert_data_multi(input, assistant_name="Assistant"):
        result_parts = []
        for entry in input:
            role = entry['role']
            assert role == 'assistant' or role == 'user'
            content = entry['content']

            if role == "assistant":
                role = assistant_name 
            elif role == "user":
                role = "User"
            else:
                raise NotImplementedError()
            result_parts.append(f"{role}: {content}")

        result_string = "\n".join(result_parts)
        return result_string

    def format_judgements(batch, optional_chat_template=None):
        # TODO expand this to include fastchat chat templates if needed
        mult_turn = True if len(batch["text_chosen"]) > 2 else False

        if mult_turn:
            prompt = None 
            answer_a =  None 
            answer_b = None 
        else:
            prompt = batch["text_chosen"][0]["content"]
            answer_a = batch["text_chosen"]
            answer_b = batch["text_rejected"]

        if model_modifier == "RM-R1-Instruct" or model_modifier == "RM-R1-Reasoning":
            name = "Chatbot"
        else:
            name = "Assistant"

        # shuffle a and b randomly for position bias
        is_shuffled = np.random.rand() > 0.5
        if is_shuffled:
            answer_a, answer_b = answer_b, answer_a
            if mult_turn: 
                if model_modifier == "RM-R1-Instruct" or model_modifier == "RM-R1-Reasoning":
                    conversation1 = convert_data_multi(batch['text_rejected'], assistant_name=f"{name} A")
                    conversation2 = convert_data_multi(batch['text_chosen'], assistant_name=f"{name} B")
                else:
                    raise NotImplementedError("Check Your mode for Multi-round")
            else:
                conversation1 = None 
                conversation2 = None 
        else:
            if mult_turn:
                if model_modifier == "RM-R1-Instruct" or model_modifier == "RM-R1-Reasoning":
                    conversation1 = convert_data_multi(batch['text_chosen'], assistant_name=f"{name} A")
                    conversation2 = convert_data_multi(batch['text_rejected'], assistant_name=f"{name} B")
                else:
                    raise NotImplementedError("Check Your mode for Multi-round")
            else:
                conversation1 = None 
                conversation2 = None 

        system_prompt, user_prompt = format_judge_answers(
            prompt, answer_a, answer_b, 
            multi_turn=mult_turn, 
            conversation1=conversation1,
            conversation2=conversation2,
            model_modifier=model_modifier,
        )

        if optional_chat_template is not None:
            raise NotImplementedError("Chat templates not implemented yet")
            optional_chat_template.set_system_message(system_prompt)
            optional_chat_template.messages = []
            optional_chat_template.append_message(optional_chat_template.roles[0], user_prompt)
            optional_chat_template.append_message(optional_chat_template.roles[1], None)
            prompt = optional_chat_template.get_prompt()
        else:
            if model_modifier == "RM-R1-Reasoning":
                messages = [
                    {'role': "user", 'content':user_prompt},
                ]
            else:
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": user_prompt},
                ]         
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # chat template already include special tokens
            # when vllm runs model.generate on prompts, the tokenizer is applied to the prompts
            # defaulting to add_special_tokens=True - this will end up duplicating the special tokens
            # so we need to tokenize without adding special tokens
            tokenized_prompt = tokenizer(prompt, add_special_tokens=False, return_length=True)
            prompt_ids = tokenized_prompt["input_ids"]

        batch["text"] = prompt
        batch["is_shuffled"] = is_shuffled
        batch["prompt_ids"] = prompt_ids
        return batch
    
    chat_template = None 
    dataset_prompts = dataset.map(format_judgements, fn_kwargs={"optional_chat_template": chat_template})
    prompts = dataset_prompts["text"]
    prompt_ids = dataset_prompts["prompt_ids"]
    is_shuffled = dataset_prompts["is_shuffled"]

    # generate
    logger.info("*** Run inference ***")

    sampling_params = SamplingParams(
        n=1,
        temperature=0,
        top_p=1,
        max_tokens=args.max_tokens,
        stop_token_ids=None,
    )
    outputs = model.generate(prompts, sampling_params=sampling_params)

    logger.info("*** Inference done ***") 
    if args.model == 'meta-llama/Llama-3.1-8B-Instruct':
        answers = outputs
    else:
        answers = [o.outputs[0].text for o in outputs]

    winners = [process_judgement(a, model_modifier) for a in answers]

    def process_shuffled(win, shuffle):
        if shuffle:
            winner_text = "B"
            loser_text = "A"
        else:
            winner_text = "A"
            loser_text = "B"

        if win == winner_text:
            return 1
        elif win == loser_text:
            return 0
        elif win == "strong_error":
            return 0 
        elif win == "error":
            return 0 # tie 
        elif win == "tie":
            return 0.5 
        else:  # if "error"
            raise NotImplementedError("Error with your output")

    results = [process_shuffled(w, s) for w, s in zip(winners, is_shuffled)]

    final_score_json = []
    final_score_json.append({
        "Absolute Accuracy (all correct / all)": sum(results) / len(results)
    })

    ############################
    # Print & process results
    ############################
    # add column for results for easy printing
    out_dataset = dataset.add_column("results", results)
    out_dataset = out_dataset.add_column("answers", answers) 
    out_dataset = out_dataset.add_column("Is_Chosen_Answer_Shuffled_toPositionB", is_shuffled)

    if args.meta_result_save_dir is not None:
        Meta_result_save_dir = Path(args.meta_result_save_dir) / args.model_save_name / "RMB" / args.model_save_meta
        Meta_result_save_dir.mkdir(exist_ok=True, parents=True)

        data_as_list = []
        for sample in out_dataset:
            data_as_list.append(sample)
        
        log_result_dir = Meta_result_save_dir / "log_result"
        log_result_dir.mkdir(exist_ok=True, parents=True)
        with open(log_result_dir / "raw_logs.json", "w") as f:
            json.dump(data_as_list, f, indent=2) 


        from collections import defaultdict

        # Assuming `dataset` is your list of data dictionaries
        grouped_by_idx = defaultdict(list)
        for item in out_dataset:
            idx = item["idx"]
            grouped_by_idx[idx].append(item)
        
        with open(log_result_dir / "group_by_same_id_logs.json", "w") as f:
            json.dump(grouped_by_idx, f, indent=2) 
        
        total_bon, total_bon_correct = 0,0 
        sub_category_dataset = []
        for _, group_result in grouped_by_idx.items():
            is_correct = True 
            for example in group_result:
                curr_correct = example["results"]
                if curr_correct == 0:
                    is_correct = False 
                elif curr_correct == 1:
                    pass 
                elif curr_correct == 0.5:
                    is_correct = False 
                else:
                    raise NotImplementedError("Check the correct")
            if is_correct:
                total_bon_correct += 1 
            total_bon += 1 
            item = {
                "category_path": group_result[0]["category_path"],
                "results": int(is_correct)
            }
            sub_category_dataset.append(item)

        final_score_json[0]["BoN Final Accuracy"] = total_bon_correct / total_bon
        final_score_json[0]["BoN Absolute Number of Correct"] = total_bon_correct 
        final_score_json[0]["BoN Absolute Total Numbers"] = total_bon

        print("BoN Final Accuracy: ", total_bon_correct / total_bon)

        sub_category_dataset = Dataset.from_list(sub_category_dataset)
        results_grouped = {} 
        present_subsets = np.unique(sub_category_dataset['category_path'])
        
        for subset in present_subsets:
            subset_dataset = sub_category_dataset.filter(lambda example: example["category_path"] == subset)
            num_correct = sum(subset_dataset["results"])
            num_total = len(subset_dataset["results"])
            print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
            results_grouped[subset] = num_correct / num_total

        final_score_json.append(results_grouped)


        pure_score_result_dir = Meta_result_save_dir / "score_result"
        pure_score_result_dir.mkdir(exist_ok=True, parents=True) 
        with open(pure_score_result_dir / "Final_score.json", "w") as f:
            json.dump(final_score_json, f, indent=4)

if __name__ == "__main__":
    main()