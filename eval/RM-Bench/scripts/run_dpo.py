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
from trl.trainer.utils import DPODataCollatorWithPadding

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.utils import convert_robust_dataset_to_preference_dataset_list, load_eval_dataset, compute_accuracy

import gc


from rewardbench import (
    DPO_MODEL_CONFIG,
    DPOInference,
    # load_eval_dataset,
    save_to_hub,
    torch_dtype_mapping,
)
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
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
    parser.add_argument("--model", type=str, required=True, help="path to model")
    parser.add_argument("--ref_model", type=str, default=None, help="path to model")
    parser.add_argument(
        "--ref_free_type", type=str, default="avg", help="type of reference free normalization (norm, avg, or sum)"
    )
    parser.add_argument("--datapath", type=str, default="data/reward-bench", help="path to data")
    parser.add_argument("--tokenizer", type=str, default=None, help="path to non-matching tokenizer")
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument("--batch_size", type=int, default=6, help="batch size for inference")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--debug", action="store_true", default=False, help="use only 10 examples")
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
    accelerator = Accelerator()

    ###############
    # Setup logging
    ###############
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

    offical_model_name = args.model.replace("RewardModels/", "")
    if args.model in DPO_MODEL_CONFIG:
        config = DPO_MODEL_CONFIG[offical_model_name]
    else:
        config = DPO_MODEL_CONFIG["default"]
    logger.info(f"Using dpo model config: {config}")

    model_builder = config["model_builder"]
    tokenizer_builder = config["tokenizer_builder"]

    # check datatype from argparse
    if args.torch_dtype == torch.bfloat16:
        logger.warning("Loading weights directly as bfloat16 for PyTorch dtype")
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    assert args.model != args.ref_model, "policy and reference model should be different"
    # load chat template
    chat_template = args.chat_template
    conv = get_conv_template(chat_template)

    # define reference free
    if args.ref_model is None:
        ref_free = True
        logger.info("Running reference free DPO - no reference model provided")
    else:
        ref_free = False
        logger.info(f"Running DPO with reference model {args.ref_model}")

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = tokenizer_builder(tokenizer_path, trust_remote_code=args.trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    # if no BOS token, set as pad token, e.g. QWEN models
    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id


    raw_dataset_list = convert_robust_dataset_to_preference_dataset_list(args.datapath)
    
    
    if (
        ("llama-3" in args.model)
        or ("Llama3" in args.model)
        or ("Llama-3" in args.model)
        or ("LLaMA3" in args.model)
        or args.not_quantized
    ):
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
        }
        model_kwargs_ref = {
            "device_map": "auto",
            "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
        }
    else:
        model_kwargs = {
            "load_in_8bit": True,
            "device_map": "auto",
            "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
        }
        model_kwargs_ref = {
            "load_in_8bit": True,
            "device_map": "auto",
            "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
        }

    model = model_builder(
        args.model,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="sdpa",
        **model_kwargs,
    )

    if ref_free:
        ref_model = None
    else:
        ref_model = model_builder(
            args.ref_model,
            trust_remote_code=args.trust_remote_code,
            **model_kwargs_ref,
        )

    # use internal inference functions in DPO trainer
    dpo = DPOInference(
        model,
        ref_model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        ref_free_norm=args.ref_free_type,
        # norm is norm, avg is average, sum is sum
    )
    # score_original = []
    score_chosen = []
    score_rejected = []
    
    for dataset_idx, raw_dataset in enumerate(raw_dataset_list):
        
        # clear cuda memory cache
        # model = None
        dataset = None
        dataloader = None
        tokenized_dataset = None
        batch = None
        # del model
        # Synchronize and clear GPU memory
        torch.cuda.synchronize()
        del dataset
        del dataloader
        del tokenized_dataset
        del batch
        gc.collect()
        torch.cuda.empty_cache()
        # gc.collect()
        # torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # torch.cuda.empty_cache()
        # prin the gpu memory usage
        
        # for device in range(torch.cuda.device_count()):
        #     cuda.select_device(device)  # Select the GPU device
        #     cuda.close()  # Clear the memory
        #     cuda.select_device(device)  # Reinitialize the GPU device if necessary
        
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB")
        
        dataset, subsets = load_eval_dataset(
            raw_dataset,
            core_set=not args.pref_sets,
            conv=conv,
            tokenizer=tokenizer,
            logger=logger,
            keep_columns=["text_chosen", "text_rejected", "id", "prompt"],
        )

        dataset = dataset.remove_columns("id")
        # debug: use only 10 examples
        if args.debug:
            dataset = dataset.select(range(10))
            subsets = subsets[:10]

        ############################
        # Load reward model pipeline
        ############################
        BATCH_SIZE = args.batch_size

        # tokenize dataset
        column_names = list(dataset.features)

        tokenized_dataset = dataset.map(dpo.tokenize_row, remove_columns=column_names)

        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=dpo.label_pad_token_id,
                is_encoder_decoder=dpo.is_encoder_decoder,
            ),
            # collate_fn = lambda x: x, # fix weird batching error
            shuffle=False,
            drop_last=False,
        )
        results = []
        scores_chosen = []
        scores_rejected = []

        for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
            logger.info(f"RM inference step {step}/{len(dataloader)}")

            rewards_chosen, rewards_rejected = dpo.inference_step(batch, ref_free=ref_free)

            # for each item in batch, record 1 if chosen > rejected
            # extra score from dict within batched results (e.g. logits)
            # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
            if isinstance(rewards_chosen[0], dict):
                scores_chosen_batch = [result["score"] for result in rewards_chosen]
                scores_rejected_batch = [result["score"] for result in rewards_rejected]
            # for classes that directly output scores (custom code)
            else:
                scores_chosen_batch = rewards_chosen.float().cpu().numpy().tolist()  # convert to float for bfloat16 case
                scores_rejected_batch = rewards_rejected.float().cpu().numpy().tolist()

            [
                results.append(1) if chosen > rejected else results.append(0)
                for chosen, rejected in zip(scores_chosen_batch, scores_rejected_batch)
            ]
            scores_chosen += scores_chosen_batch
            scores_rejected += scores_rejected_batch


        score_chosen.append(scores_chosen)
        score_rejected.append(scores_rejected)
            
    
    ############################
    # Save results
    ############################
    
    import json
    # HACK: load the dataset from the file
    dataset_json:list = json.load(open(args.datapath))
    

    print(f"Type of score_chosen: {type(score_chosen[0])}")
    print(f"Lenght of score_chosen: {len(score_chosen[0])}")
    # print(score_chosen[0])
    print(f"Type of score_rejected: {type(score_rejected[0])}")
    print(f"Lenght of score_rejected: {len(score_rejected[0])}")
    # print(score_rejected[0])
    
    for idx, unit in enumerate(dataset_json):
        unit['score_chosen'] = [
            score_list[idx] for score_list in score_chosen
        ]
        unit['score_rejected'] = [
            score_list[idx] for score_list in score_rejected
        ]
    
    # save to results folder with the name + model name + timestamp
    filename = os.path.basename(args.datapath).replace(".json", "")
    model_name = args.model.split("/")[-1]
    ref_model_name = args.ref_model.split("/")[-1] if args.ref_model else "ref_free"
    output_dir = f"results/DPO/{offical_model_name}"
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