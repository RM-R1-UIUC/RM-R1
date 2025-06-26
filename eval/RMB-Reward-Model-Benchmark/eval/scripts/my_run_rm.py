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

import argparse
import logging
import os
import sys
import json
import yaml
import pandas as pd

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from fastchat.conversation import get_conv_template
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from rewardbench import (
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
    save_to_hub,
)
from my_utils.utils import torch_dtype_mapping, load_eval_dataset
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section

from my_utils.armorm import ArmoRMPipeline

# Enable TensorFloat32 (TF32) tensor cores on Ampere GPUs for matrix multiplications (faster than FP32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="", required=True, help="model root")
    parser.add_argument("--model", type=str, required=True, help="path to model")
    parser.add_argument("--tokenizer", type=str, default=None, help="path to non-matching tokenizer to model")
    parser.add_argument("--dataset_dir", type=str, required=True, help="path to data_dir")
    parser.add_argument("--single_data", type=bool, required=True, help="if single")
    parser.add_argument("--dataset", type=str, required=True, help="path to data")
    parser.add_argument("--results", type=str, required=True, help="path to results")
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for inference")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length of RM inputs (passed to pipeline)")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--debug", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--disable_beaker_save", action="store_true", help="disable saving the main results in a file for AI2 Beaker"
    )
    parser.add_argument(
        "--not_quantized", action="store_true", help="disable quantization for models that are quantized by default"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32", "float64"],
        help="PyTorch dtype (default: float16)",
    )
    args = parser.parse_args()
    args.torch_dtype = torch_dtype_mapping(args.torch_dtype)
    return args

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_parameters(config, key):
    if key in config:
        return config[key]
    else:
        raise KeyError(f"Configuration for '{key}' not found.")

# find all dataset 
def find_json_files(directory):
    json_files = []
    
    # 遍历目录及其子目录
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    return json_files

    

def main():
    args = get_args()
    ###############
    # Setup logging
    ###############
    accelerator = Accelerator()
    current_device = accelerator.process_index

    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    model_dir = args.model_dir
    if model_dir == "":
        p2n = {
            'ArmoRM-Llama3-8B-v0.1': 'RLHFlow/ArmoRM-Llama3-8B-v0.1',
            'Eurus-RM-7b': 'openbmb/Eurus-RM-7b',
            'stablelm-2-12b-chat': 'stabilityai/stablelm-2-12b-chat',
            'Starling-RM-34B': 'Nexusflow/Starling-RM-34B',
            'internlm2-7b-reward': 'internlm/internlm2-7b-reward',
            'internlm2-20b-reward': 'internlm/internlm2-20b-reward',
            'Llama3-70B-SteerLM-RM': 'nvidia/Llama3-70B-SteerLM-RM',
            'tulu-v2.5-13b-preference-mix-rm': 'allenai/tulu-v2.5-13b-preference-mix-rm'
        }
    else:
        p2n = {
            model_dir + '/ArmoRM-Llama3-8B-v0.1': 'RLHFlow/ArmoRM-Llama3-8B-v0.1',
            model_dir + '/Eurus-RM-7b': 'openbmb/Eurus-RM-7b',
            model_dir + '/stablelm-2-12b-chat': 'stabilityai/stablelm-2-12b-chat',
            model_dir + '/Starling-RM-34B': 'Nexusflow/Starling-RM-34B',
            model_dir + '/internlm2-7b-reward': 'internlm/internlm2-7b-reward',
            model_dir + '/internlm2-20b-reward': 'internlm/internlm2-20b-reward',
            model_dir + '/Llama3-70B-SteerLM-RM': 'nvidia/Llama3-70B-SteerLM-RM',
            model_dir + '/tulu-v2.5-13b-preference-mix-rm': 'allenai/tulu-v2.5-13b-preference-mix-rm'
        }

    model_config = load_config('../RMB-Reward-Model-Benchmark/eval/scripts/configs/eval_configs.yaml')
    if model_dir == "":
        model_name = args.model
    else:
        model_name = p2n[args.model]
    config_dict = get_parameters(model_config, model_name)
    trust_remote_code = config_dict['trust_remote_code']

    # load chat template
    # default: tulu
    if 'chat_template' in config_dict.keys():
        chat_template = config_dict['chat_template']
    else:
        chat_template = "tulu"
    try:
        conv = get_conv_template(chat_template)
    except Exception as e:
        conv = get_conv_template("tulu")
    
    logger.info(f"Running reward model on {args.model} with chat template {chat_template}")

    # load reward model
    if trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    if model_name in REWARD_MODEL_CONFIG:
        config = REWARD_MODEL_CONFIG[model_name]
    else:
        config = REWARD_MODEL_CONFIG["default"]
    logger.info(f"Using reward model config: {config}")

    quantized = config["quantized"]  # only Starling isn't quantized for now
    # if llama-3 in name, switch quantized to False (severely degrades performance)
    if (
        ("llama-3" in args.model)
        or ("Llama3" in args.model)
        or ("Llama-3" in args.model)
        or ("LLaMA3" in args.model)
        or ("llama3" in args.model)
        or args.not_quantized
    ):
        quantized = False
        logger.info(f"Disabling quantization for llama-3 or override flag (--not_quantized: {args.not_quantized})")

    custom_dialogue = config["custom_dialogue"]
    model_type = config["model_type"]
    print("model_type: ", model_type)
    model_builder = config["model_builder"]
    if model_name == 'RLHFlow/ArmoRM-Llama3-8B-v0.1':
        pipeline_builder = ArmoRMPipeline
    else:
        pipeline_builder = config["pipeline_builder"]
    torch_dtype = config.get("torch_dtype", None)
    # if not datatype in config (default), check args
    if torch_dtype is None:
        # if datatype is bfloat16, then manually turn off quantizaiton (done with bitsandbytes)
        if args.torch_dtype == torch.bfloat16:
            quantized = False
            logger.info("Disabling quantization for bfloat16 datatype")
        torch_dtype = args.torch_dtype

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    # tokenizer_path = args.tokenizer if args.tokenizer else args.model
    if model_name != config_dict['tokenizer']:
        tokenizer_path = config_dict['tokenizer']
    else:
        tokenizer_path = args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
    if not custom_dialogue:  # not needed for PairRM / SteamSHP
        tokenizer.truncation_side = "left"  # copied from Starling, but few samples are above context length
    if args.single_data:
        data_path_list = [args.dataset]
    else:
        data_path_list = find_json_files(args.dataset_dir)

    print(args.dataset)
    dataset, subsets = load_eval_dataset(
        core_set=False,
        EXTRA_PREF_SETS = data_path_list,
        conv=conv,
        custom_dialogue_formatting=custom_dialogue,
        tokenizer=tokenizer,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "pair_uid", "category_path"],
    )

    # copy id for saving, then remove
    ids = dataset["pair_uid"]
    # dataset = dataset.remove_columns("pair_uid")

    # debug: use only 10 examples
    if args.debug:
        dataset = dataset.select(range(10))
        subsets = subsets[:10]
        ids = ids[:10]

    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = config_dict['batch_size']
    logger.info("*** Load reward model ***")
    reward_pipeline_kwargs = {
        "batch_size": BATCH_SIZE,  # eval_args.inference_batch_size,
        "truncation": True,
        "padding": True,
        "max_length": args.max_length,
        "function_to_apply": "none",  # Compute raw logits
        "return_token_type_ids": False,
    }
    if quantized:
        model_kwargs = {
            "load_in_8bit": False,
            "device_map": {"": current_device},
            "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
        }
    else:
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch_dtype,
        }

    model = model_builder(args.model, **model_kwargs, trust_remote_code=trust_remote_code)
    reward_pipe = pipeline_builder(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
    )

    ############################
    # Tokenization settings & dataset preparation
    ############################
    # set pad token to eos token if not set
    if reward_pipe.tokenizer.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
        reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id
    # For models whose config did not contains `pad_token_id`
    if reward_pipe.model.config.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.pad_token_id

    # if using fastchat template (no template in tokenizer), make the RM tokenizer output an EOS token
    if not check_tokenizer_chat_template(tokenizer):
        reward_pipe.tokenizer.add_eos_token = True

    ############################
    # Run inference [1/2]" built in transformers
    ############################
    # if using HF pipeline, can pass entire dataset and get results
    # first, handle custom pipelines that we must batch normally
    if pipeline_builder == pipeline:
        logger.info("*** Running forward pass via built in pipeline abstraction ***")
        # this setup can be optimized slightly with one pipeline call
        # prepare for inference
        reward_pipe = accelerator.prepare(reward_pipe)

        results_rej = reward_pipe(dataset["text_rejected"], **reward_pipeline_kwargs)
        results_cho = reward_pipe(dataset["text_chosen"], **reward_pipeline_kwargs)

        # extract scores from results which is list of dicts, e.g. [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
        scores_chosen = [result["score"] for result in results_cho]
        scores_rejected = [result["score"] for result in results_rej]

        # pairwise comparison list comprehension
        results = [1 if chosen > rejected else 0 for chosen, rejected in zip(scores_chosen, scores_rejected)]

    ############################
    # Run inference [2/2] custom pipelines
    ############################
    else:
        logger.info("*** Running dataloader to collect results ***")
        # TODO make more custom pipelines work with pre-tokenized data
        from torch.utils.data.dataloader import default_collate

        # for PairRM, hmm, will move all of this later
        def custom_collate_fn(batch):
            # check if ['text_chosen'] is in first batch element
            # Check if the first element of the batch is a dictionary
            if isinstance(batch[0]["text_chosen"][0], dict):
                return batch  # Return the batch as-is if it's a list of dicts
            else:
                return default_collate(batch)  # Use the default collate behavior otherwise

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            collate_fn=custom_collate_fn,  # if not args.pref_sets else None,
            shuffle=False,
            drop_last=False,
        )

        dataloader, model = accelerator.prepare(dataloader, reward_pipe.model)
        reward_pipe.model = model

        results = []
        scores_chosen = []
        scores_rejected = []
        for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
            logger.info(f"RM inference step {step}/{len(dataloader)}")

            if model_type == "Custom Classifier":
                # print("Custom Classifier")
                text_rejected = [b["text_rejected"] for b in batch]
                text_chosen = [b["text_chosen"] for b in batch]
                results_sub = reward_pipe(text_chosen, text_rejected, **reward_pipeline_kwargs)
                if model_name == 'RLHFlow/ArmoRM-Llama3-8B-v0.1':
                    score_chosen_batch = [result[0] for result in results_sub]
                    score_rejected_batch = [result[1] for result in results_sub]
                    [results.append(1) if result[0] > result[1] else results.append(0) for result in results_sub]
                    scores_chosen.extend(score_chosen_batch)
                    scores_rejected.extend(score_rejected_batch)
                else:
                    [results.append(1) if result else results.append(0) for result in results_sub.cpu().numpy().tolist()]
                    scores_chosen.extend([None] * len(results_sub))
                    scores_rejected.extend([None] * len(results_sub))
                # [results.append(1) if result else results.append(0) for result in results_sub.cpu().numpy().tolist()]
            else:
                # print("other Classifier")
                rewards_chosen = reward_pipe(batch["text_chosen"], **reward_pipeline_kwargs)
                rewards_rejected = reward_pipe(batch["text_rejected"], **reward_pipeline_kwargs)

                # for each item in batch, record 1 if chosen > rejected
                # extra score from dict within batched results (e.g. logits)
                # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
                if isinstance(rewards_chosen[0], dict):
                    score_chosen_batch = [result["score"] for result in rewards_chosen]
                    score_rejected_batch = [result["score"] for result in rewards_rejected]
                # for classes that directly output scores (custom code)
                else:
                    score_chosen_batch = (
                        rewards_chosen.float().cpu().numpy().tolist()
                    )  # cast to float in case of bfloat16
                    score_rejected_batch = rewards_rejected.float().cpu().numpy().tolist()

                # log results
                [
                    results.append(1) if chosen > rejected else results.append(0)
                    for chosen, rejected in zip(score_chosen_batch, score_rejected_batch)
                ]
                scores_chosen.extend(score_chosen_batch)
                scores_rejected.extend(score_rejected_batch)
    
    # dataset.remove_columns("text_chosen")
    # dataset.remove_columns("text_rejected")
    def add_feature(dataset):
        dataset_json = []
        for i in range(len(dataset)):
            data_dict = dataset[i]
            data_dict.pop("text_chosen")
            data_dict.pop("text_rejected")
            data_dict["chosen_reward"] = scores_chosen[i]
            data_dict["reject_reward"] = scores_rejected[i]
            data_dict["is_correct"] = results[i]
            dataset_json.append(data_dict)
    
        return dataset_json
    # dataset["chosen_reward"] = scores_chosen
    # dataset["reject_reward"] = scores_rejected
    # dataset["is_correct"] = results
    dataset_json = add_feature(dataset)
    model_acc = sum(results) / len(results)
    model_n = args.model.split("/")[-1]
    print(model_n, " ACC: ", model_acc)
    with open(args.results, 'w', encoding='utf-8') as file:
        json.dump(dataset_json, file, indent=2, ensure_ascii=False)
        print(args.results, "write down results")


    def rm_rank(r_dict):
        r_list = []
        for r_key in r_dict.keys():
            m_dict = r_dict[r_key]
            new_dict = {}
            new_dict["rm_name"] = r_key
            new_dict["ACC"] = m_dict["ACC"]
            r_list.append(new_dict)
        sorted_r_list = sorted(r_list, key=lambda x: x['ACC'], reverse=True)
        sorted_r_dict = {}
        r_n = 1
        for sorted_m_dict in sorted_r_list:
            sorted_r_dict[sorted_m_dict["rm_name"]] = {
                "ACC": sorted_m_dict["ACC"],
                "rank": r_n
            }
            r_n += 1

        return sorted_r_dict
    
    all_results_path = 'rm_rank.json'
    if os.path.exists(all_results_path):
        with open(all_results_path, 'r', encoding = 'utf-8') as input_file:
            rm_rank_dict = json.load(input_file)
        if model_n in rm_rank_dict.keys():
            rm_rank_dict[model_n]["ACC"] = model_acc
        else:
            rm_rank_dict[model_n] = {}
            rm_rank_dict[model_n]["ACC"] = model_acc
        n_rm_rank_dict = rm_rank(rm_rank_dict)
    else:
        with open(all_results_path, 'r', encoding = 'utf-8') as input_file:
            pass
        n_rm_rank_dict = {}
        n_rm_rank_dict[model_n] = {
            "ACC": model_acc,
            "rank": 1
        }
    
    print("rank_dict: ", n_rm_rank_dict)
    
    with open(all_results_path, 'w', encoding='utf-8') as file:
        json.dump(n_rm_rank_dict, file, indent=2, ensure_ascii=False)
        print(all_results_path, "write down all_result_path")
    # dataset.to_json(args.results)

    # df = pd.read_csv(args.dataset)

    # # 新增两列，列名为'chose'和'rejected'
    # df['chosen_reward'] = scores_chosen
    # df['rejected_reward'] = scores_rejected
    # df['is_correct'] = results
    # ACC = sum(results) / len(results)
    # new_row = {'id': len(results), 'prompt': len(results), 'subset': len(results), 'chosen': len(results), 'rejected': len(results), 'chosen_reward': sum(results), 'rejected_reward': len(results), 'is_correct': ACC}
    # df.loc[len(df)] = new_row

    # # 保存为新文件
    # df.to_csv(args.results, index=False)

if __name__ == "__main__":
    main()
