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

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from fastchat.conversation import get_conv_template
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.utils import convert_robust_dataset_to_preference_dataset_list, load_eval_dataset, compute_accuracy

import gc

from rewardbench import (
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
    # load_eval_dataset,
    save_to_hub,
    torch_dtype_mapping,
)
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section

# Enable TensorFloat32 (TF32) tensor cores on Ampere GPUs for matrix multiplications (faster than FP32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
    parser.add_argument("--model", type=str, required=True, help="path to model")
    parser.add_argument("--tokenizer", type=str, default=None, help="path to non-matching tokenizer to model")
    # parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument("--chat_template", type=str, default=None, help="path to chat template")
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--datapath", type=str, default="data/reward-bench", help="path to data")
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

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")
    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    # load chat template
    chat_template = args.chat_template
    if chat_template is not None:
        conv = get_conv_template(chat_template)
    else:
        conv = None 
    logger.info(f"Using conversation template {chat_template}: {conv}")
    
    offical_model_name = args.model.replace("RewardModels/", "")
    if offical_model_name in REWARD_MODEL_CONFIG:
        # delete the "RewardModel/" prefix
        config = REWARD_MODEL_CONFIG[offical_model_name]
    else:
        config = REWARD_MODEL_CONFIG["default"]
    logger.info(f"Using reward model config: {config}")

    # Default entries
    # "model_builder": AutoModelForSequenceClassification.from_pretrained,
    # "pipeline_builder": pipeline,
    # "quantized": True,
    # "custom_dialogue": False,
    # "model_type": "Seq. Classifier"

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
    if model_type == "Custom Classifier":
        raise  NotImplementedError("For the Custom Classifier model like NVIDIA SteerLM, plz refer to the NVIDIA original code")
    model_builder = config["model_builder"]
    pipeline_builder = config["pipeline_builder"]
    torch_dtype = config.get("torch_dtype", None)
    # if not datatype in config (default), check args
    if torch_dtype is None:
        # if datatype is bfloat16, then manually turn off quantizaiton (done with bitsandbytes)
        if args.torch_dtype == torch.bfloat16:
            quantized = False
            logger.info("Disabling quantization for bfloat16 datatype")
        torch_dtype = args.torch_dtype

    # not included in config to make user explicitly understand they are passing this
    trust_remote_code = args.trust_remote_code

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if not custom_dialogue:  # not needed for PairRM / SteamSHP
        tokenizer.truncation_side = "left"  # copied from Starling, but few samples are above context length
    
    
    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size
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
            "load_in_8bit": True,
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
    
    
    raw_dataset_list = convert_robust_dataset_to_preference_dataset_list(args.datapath)
    
    # print(raw_dataset_list)
    # print(raw_dataset_list[0])
    # print(raw_dataset_list[0][0])
    # exit()

    # score_original = []
    score_chosen = []
    score_rejected = []
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
            conv=conv,
            custom_dialogue_formatting=custom_dialogue,
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
            unit_score_chosen_list = [result["score"] for result in results_cho]
            unit_score_rejected_list = [result["score"] for result in results_rej]

            # pairwise comparison list comprehension
            results = [1 if chosen > rejected else 0 for chosen, rejected in zip(unit_score_chosen_list, unit_score_rejected_list)]

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
            unit_score_chosen_list = []
            unit_score_rejected_list = []
            for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
                # logger.info(f"RM inference step {step}/{len(dataloader)}")

                if model_type == "Custom Classifier":
                    raise NotImplementedError("For the Custom Classifier model like NVIDIA SteerLM, plz refer to the NVIDIA original code")
                else:
                    # print(batch)
                    # exit()
                    rewards_chosen = reward_pipe(batch["text_chosen"], **reward_pipeline_kwargs)
                    rewards_rejected = reward_pipe(batch["text_rejected"], **reward_pipeline_kwargs)
                    print(f"rewards_chosen: {rewards_chosen}")
                    print(f"rewards_rejected: {rewards_rejected}")
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
                    for chosen, rejected in zip(score_chosen_batch, score_rejected_batch):
                        print(f"chosen: {chosen}, rejected: {rejected}")
                        if chosen > rejected:
                            results.append(1)
                        else:
                            results.append(0)
                    unit_score_chosen_list.extend(score_chosen_batch)
                    unit_score_rejected_list.extend(score_rejected_batch)



        score_chosen.append(unit_score_chosen_list)
        score_rejected.append(unit_score_rejected_list)

    
    ############################
    # Save results
    ############################
    
    import json
    # HACK: load the dataset from the file
    dataset_json:list = json.load(open(args.datapath))
    
    # print(f"Type of score_original: {type(score_original)}")
    # print(f"Lenght of score_original: {len(score_original)}")
    # print(score_original)
    print(f"Type of score_chosen: {type(score_chosen[0])}")
    print(f"Lenght of score_chosen: {len(score_chosen[0])}")
    # print(score_chosen[0])
    print(f"Type of score_rejected: {type(score_rejected[0])}")
    print(f"Lenght of score_rejected: {len(score_rejected[0])}")
    # print(score_rejected[0])
    
    for idx, unit in enumerate(dataset_json):
        # unit['score_orig'] = score_original[idx]
        unit['score_chosen'] = [
            score_list[idx] for score_list in score_chosen
        ]
        unit['score_rejected'] = [
            score_list[idx] for score_list in score_rejected
        ]
        # if all the elemnts in the list are list and all the elements is of length 1
        if all(isinstance(elem, list) and len(elem) == 1 for elem in unit['score_chosen']):
            unit['score_chosen'] = [elem[0] for elem in unit['score_chosen']]
        if all(isinstance(elem, list) and len(elem) == 1 for elem in unit['score_rejected']):
            unit['score_rejected'] = [elem[0] for elem in unit['score_rejected']]
    
    # save to results folder with the name + model name + timestamp
    filename = os.path.basename(args.datapath).replace(".json", "")
    model_name = args.model.split("/")[-1]
    ref_model_name = "REWORD_MODEL"
    # make a dir at results with official model name
    output_dir = f"results/Seq_Classifier/{offical_model_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    from datetime import datetime
    output_path = os.path.join(output_dir, f"{filename}_{model_name}_{ref_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_path, "w") as f:
        json.dump(dataset_json, f, indent=4, ensure_ascii=False)
    
        acc_dict = compute_accuracy(dataset_json)
    print(f"The accuracy of model {model_name}\n in the dataset {filename} is:\n {acc_dict}")

if __name__ == "__main__":
    main()