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
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch 

import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from fastchat.conversation import get_conv_template

from pathlib import Path
import openai 
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.utils import convert_robust_dataset_to_preference_dataset_list, load_eval_dataset, compute_accuracy, compute_accuracy_gen
import gc



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

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",  # allow list of models (ensemble)
        required=True,
        help="name of OpenAI model to use (TODO add more providers/models)",
    )
    parser.add_argument("--chat_template", type=str, default=None, help="fastchat chat template (optional)")
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus to use, for multi-node vllm")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.9, help="gpu utilization for vllm")
    # parser.add_argument("--vllm_max_seq_length", type=int, default=None, help="max sequence length for vllm")
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--debug", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--num_threads", type=int, default=10, help="number of threads to use for parallel processing of examples"
    )
    parser.add_argument(
        "--disable_beaker_save", action="store_true", help="disable saving the main results in a file for AI2 Beaker"
    )
    parser.add_argument(
        "--force_local", action="store_true", default=False, help="force local run, even if model is on Together API"
    )
    parser.add_argument(
        "--model_save_name", default="default_save", type=str
    )
    parser.add_argument(
        '--max_tokens', type=int, default=2048, help='max tokens for generation'
    )
    parser.add_argument(
        "--META_RESULT_SAVE_DIR", type=str, default="results/Gen_RMs", help="path to save the results"
    )
    parser.add_argument("--datapath", type=str, default="data/reward-bench", help="path to data")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
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

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")

    model_type = "Generative RM"

    # if model is list, make type + PoLL and check multiple is odd
    if isinstance(args.model, list) and len(args.model) == 1:
        args.model = args.model[0]
    elif isinstance(args.model, list):
        model_type += " PoLL"
        # assert that is odd and > 1
        assert len(args.model) % 2 == 1

    # define variable if is API or local
    if args.force_local:
        is_api_models = False
    else:
        is_api_models = isinstance(args.model, list) or args.model in API_MODEL_LIST

    # if model isn't API, load via vllm
    if not is_api_models:
        # if multi gpu, set multiproc method to spawn
        if args.num_gpus > 1:
            # Set the environment variable
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # if args.model == 'meta-llama/Llama-3.1-8B-Instruct':
        #     from transformers import AutoModelForCausalLM
        #     model = AutoModelForCausalLM.from_pretrained(
        #         args.model,
        #         attn_implementation
        #     )
        #     model = model.cuda()
        # else:
        model = LLM(
            args.model,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=args.num_gpus,
            gpu_memory_utilization=args.vllm_gpu_util,
            # max_seq_length=args.vllm_max_seq_length,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if "Llama-3" in args.model or "llama3-8b" in args.model and "3.1" not in args.model:
            stop_token_ids = [128009]
        else:
            stop_token_ids = None
    else:
        tokenizer = None 

    # handle off-case models
    # use different prompt for prometheus/gemini models
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
    # Load dataset
    ############################

    raw_dataset_list = convert_robust_dataset_to_preference_dataset_list(args.datapath) 
    META_RESULTS_LIST = []
    META_OUTPUT_LIST = []
    META_SHUFFLED = []

    logger.info("*** Load dataset ***")

    for dataset_idx, raw_dataset in enumerate(raw_dataset_list):
    
        # clear cuda memory cache
        dataset = None
        dataloader = None
        torch.cuda.synchronize()
        del dataset
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # prin the gpu memory usage
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB")
        
        # for device in range(torch.cuda.device_count()):
        #     cuda.select_device(device)  # Select the GPU device
        #     cuda.close()  # Clear the memory
        #     cuda.select_device(device)  # Reinitialize the GPU device if necessary
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB")


        dataset, subsets = load_eval_dataset(
            raw_dataset,
            core_set=not args.pref_sets,
            conv=get_conv_template("raw"),
            custom_dialogue_formatting=True,
            tokenizer=tokenizer,
            logger=logger,
            keep_columns=["text_chosen", "text_rejected", "id"],
        )
        # copy id for saving, then remove
        ids = dataset["id"]
        dataset = dataset.remove_columns("id")

        # debug: use only 10 examples
        if args.debug:
            dataset = dataset.select(range(10))
            subsets = subsets[:10]
            ids = ids[:10]
        

        if is_api_models:
        ############################
        # Run inference via API
        ############################
            def update_progress_bar(done, total):
                # Simple text-based progress bar
                progress = int(50 * done / total)  # Calculate progress (50 chars width)
                sys.stdout.write("\r[{}{}] {}/{}".format("#" * progress, "." * (50 - progress), done, total))
                sys.stdout.flush()

            def get_judgement(batch, debug=args.debug):
                mult_turn = True if len(batch["text_chosen"]) > 2 else False
                prompt = batch["text_chosen"][0]["content"]
                answer_a = batch["text_chosen"]
                answer_b = batch["text_rejected"]

                # shuffle a and b randomly for position bias
                is_shuffled = np.random.rand() > 0.5
                if is_shuffled:
                    answer_a, answer_b = answer_b, answer_a
                    winner_text = "B"
                    loser_text = "A"
                else:
                    winner_text = "A"
                    loser_text = "B"

                if len(batch["text_chosen"]) <= 4:  # set up only for 1 or 2 turns
                    winner, request, judgement = run_judge_pair(
                        prompt, answer_a, answer_b, args.model, multi_turn=mult_turn, model_modifier=model_modifier
                    )
                    if debug:
                        print(f"Prompt: {request}")
                        print(f"Judgement: {judgement}")

                    # handle voting
                    if isinstance(winner, list):
                        # print votes if debug
                        if debug:
                            print(winner)
                        winner = max(set(winner), key=winner.count)

                    if winner == winner_text:
                        return 1, judgement, is_shuffled
                    elif winner == loser_text:
                        return 0, judgement, is_shuffled
                    elif winner == "error":
                        return 0, judgement, is_shuffled 
                    elif winner == "strong_error":
                        return 0, judgement, is_shuffled
                    elif winner == "tie":
                        return 0.5, judgement, is_shuffled 
                    else:  # if "error"
                        raise NotImplementedError("Error")  # effectively a tie
                else:
                    raise NotImplementedError("Multi Turn check!")
                    # return 0.5

            with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                # Map 'my_function' across the vector, executing in parallel using threads
                # results = list(executor.map(get_judgement, dataset))

                # Progress bar version
                results = [None] * len(dataset)  # Preallocate results list
                answers_set = [None] * len(dataset) 
                is_shuffled_set = [None] * len(dataset)
                done_tasks = 0  # Counter for completed tasks

                with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                    # Submit all tasks and hold their futures in a list
                    future_to_index = {executor.submit(get_judgement, x): i for i, x in enumerate(dataset)}

                    # As tasks complete, update progress and store results in the original order
                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        out_tuple_result = future.result()
                        curr_res, curr_ans, curr_shuff = out_tuple_result 
                        results[index] = curr_res 
                        answers_set[index] = curr_ans 
                        is_shuffled_set[index] = curr_shuff
                        # results[index] = future.result()
                        done_tasks += 1
                        update_progress_bar(done_tasks, len(dataset))
                META_RESULTS_LIST.append(results)
                META_OUTPUT_LIST.append(answers_set)
                META_SHUFFLED.append(is_shuffled_set)
                # Print newline after progress bar
                print()
        else:
            
            ############################
            # Run model weights with vllm
            ############################

            def format_judgements(batch, optional_chat_template=None):
                # TODO expand this to include fastchat chat templates if needed
                mult_turn = True if len(batch["text_chosen"]) > 2 else False

                if mult_turn:
                    print("Multi turn!")
                    exit()
                prompt = batch["text_chosen"][0]["content"]
                answer_a = batch["text_chosen"]
                answer_b = batch["text_rejected"]

                # shuffle a and b randomly for position bias
                is_shuffled = np.random.rand() > 0.5
                if is_shuffled:
                    answer_a, answer_b = answer_b, answer_a

                system_prompt, user_prompt = format_judge_answers(
                    prompt, answer_a, answer_b, multi_turn=mult_turn, model_modifier=model_modifier,
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

            # format the dataset for the model, with optional fastchat templating
            if args.chat_template is not None:
                chat_template = get_conv_template(args.chat_template)
            else:
                chat_template = None
            
            # print("Before: ", dataset[0])
            dataset_prompts = dataset.map(format_judgements, fn_kwargs={"optional_chat_template": chat_template})
            # print("#" * 100)
            # print("After: ", tokenizer.batch_decode(dataset_prompts[0]['prompt_ids']))
            # exit()


            # collect texts of dataset in list
            prompts = dataset_prompts["text"]
            prompt_ids = dataset_prompts["prompt_ids"]
            is_shuffled = dataset_prompts["is_shuffled"]

            # generate
            logger.info("*** Run inference ***")

            if model_modifier == "Atla":
                logger.info("Using Atla model for inference")
                outputs = model.generate(prompt_token_ids=prompt_ids, sampling_params=sampling_params)
            else:

                sampling_params = SamplingParams(
                    n=1,
                    temperature=0,
                    top_p=1,
                    max_tokens=args.max_tokens,
                    stop_token_ids=stop_token_ids,
                )
                outputs = model.generate(prompts, sampling_params=sampling_params)

            # print(output)
            # print("model_modifer: ", model_modifier)
            # exit()
            logger.info("*** Inference done ***")

            if args.model == 'meta-llama/Llama-3.1-8B-Instruct':
                answers = outputs
            else:
                answers = [o.outputs[0].text for o in outputs]
            # print(answers)
            winners = [process_judgement(a, model_modifier) for a in answers]
            # print("winners: ", winners) 

            def process_shuffled(win, shuffle):
                if shuffle:
                    winner_text = "B"
                    loser_text = "A"
                else:
                    winner_text = "A"
                    loser_text = "B"

                if win == winner_text:
                    return 1
                elif win == "tie":
                    return 0.5 
                else:
                    return 0
            results = [process_shuffled(w, s) for w, s in zip(winners, is_shuffled)]
            META_RESULTS_LIST.append(results)
            META_OUTPUT_LIST.append(answers)
            META_SHUFFLED.append(is_shuffled)
    
    ############################
    # Save results
    ############################
    
    import json
    # HACK: load the dataset from the file
    dataset_json:list = json.load(open(args.datapath))
    if args.debug:
        dataset_json = dataset_json[:10]
    
    for idx, unit in enumerate(dataset_json):
        unit['result'] = [
            res_list[idx] for res_list in META_RESULTS_LIST
        ]
        unit['output'] = [
            output_list[idx] for output_list in META_OUTPUT_LIST
        ]
        unit['Is_Chosen_Answer_Shuffled_toPositionB'] = [
            shuffle_list[idx] for shuffle_list in META_SHUFFLED
        ]
    
    # save to results folder with the name + model name + timestamp
    filename = os.path.basename(args.datapath).replace(".json", "")
    model_name = args.model.split("/")[-1]
    
    # make a dir at results with official model name
    if args.META_RESULT_SAVE_DIR is None: 
        output_dir = f"results/Gen_RMs/{args.model_save_name}"
    else:
        output_dir = f"{args.META_RESULT_SAVE_DIR}/{args.model_save_name}/RM-Bench/logs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"{filename}_{model_name}.json")
    with open(output_path, "w") as f:
        json.dump(dataset_json, f, indent=4, ensure_ascii=False)

    acc_dict = compute_accuracy_gen(dataset_json)
    print(f"The accuracy of model {model_name}\n in the dataset {filename} is:\n {acc_dict}")

    right, total = 0, 0 
    for item in dataset_json:
        total += len(item['result']) 
        right += sum(item['result'])

    print(f"Finished. The ordinary accuracy is: {right / total}")

if __name__ == "__main__":
    main()
