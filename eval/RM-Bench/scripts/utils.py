import logging
from fastchat.conversation import Conversation
from datasets import Dataset, DatasetDict, Value, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizer
from rewardbench.utils import check_tokenizer_chat_template, prepare_dialogue, prepare_dialogue_from_tokenizer
import numpy as np
from typing import List, Dict, Any
import json
from datasets import Dataset, load_from_disk
EXTRA_PREF_SETS = "allenai/pref-test-sets"
def convert_robust_dataset_to_preference_dataset_list(robust_dataset_path: str) -> List[Dataset]:
    # with open(robust_dataset_path, 'r') as f:
    #     robust_dataset = json.load(f)

    robust_dataset = json.load(open(robust_dataset_path))
    # Prepare the chosen and rejected dataset list
    para_corp_dataset_list = []
    num_pairs = len(robust_dataset[0]['chosen'])
    
    assert num_pairs == len(robust_dataset[0]['rejected']), \
        "The number of chosen and rejected pairs should be the same."
    
    for idx in range(num_pairs):
        para_corp_dataset = Dataset.from_dict({
            "id": [unit['id'] for unit in robust_dataset],
            "subset": ['subset' for unit in robust_dataset],
            "prompt": [unit['prompt'] for unit in robust_dataset],
            "chosen": [unit['chosen'][idx] for unit in robust_dataset],
            "chosen_model": ["chosen" for _ in robust_dataset],
            "rejected": [unit['rejected'][idx] for unit in robust_dataset],
            "rejected_model": ["rejected" for _ in robust_dataset],
        })
        para_corp_dataset_list.append(para_corp_dataset)

    return para_corp_dataset_list

def split_dataset_by_domain(dataset: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    domains = ["chat","math","code","safety"]
    domain_dataset_dict = {}
    for domain in domains:
        domain_dataset_dict[domain] = [example for example in dataset if example['domain'].startswith(domain)]
    
    # pop the domain keys
    for domain in domain_dataset_dict:
        for example in domain_dataset_dict[domain]:
            example.pop('domain')
    
    return domain_dataset_dict


def compute_accuracy_gen(results: List[Dict[str, Any]]) -> Dict[str, float]:
    if 'domain' in results[0]:
        # this indicates this is total_dataset.json
        print('We are handling total_dataset.json')
        print('Splitting the dataset by domain...')
        # thus we need to split the results into different domains
        split_results = split_dataset_by_domain(results)
        domain_results = {}
        for domain in split_results:
            domain_results[domain] = compute_accuracy_gen(split_results[domain])
        domain_avg_results = {}
        for domain in domain_results:
            domain_avg_results[domain] = np.mean(list(domain_results[domain].values()))
        domain_hard_normal_easy_acc = {
            "hard_acc": np.mean([domain_results[domain]["hard_acc"] for domain in domain_results]),
            "normal_acc": np.mean([domain_results[domain]["normal_acc"] for domain in domain_results]),
            "easy_acc": np.mean([domain_results[domain]["easy_acc"] for domain in domain_results])
        }
        total_avg_acc = np.mean([domain_avg_results[domain] for domain in domain_avg_results])
        # merge the results into one falten dictionary
        final_results = {}
        # merge domain_avg_results into final_results
        final_results.update(domain_avg_results)
        # merge domain_hard_normal_easy_acc into final_results
        final_results.update(domain_hard_normal_easy_acc)
        # merge total_avg_acc into final_results
        final_results.update({"total_avg_acc": total_avg_acc})
        return final_results
            
    
    # results is a list of dictionaries, each dictionary contains the following keys:
    # score_chosen: [float, float, float], the scores of the chosen responses
    # score_rejected: [float, float, float], the scores of the rejected responses
    # the scores are in the order of [concise, detailed_plain, detailed_markdown]
    # we will compare the scores of chosen responses and rejected responses iteratively
    # formatted as a 3x3 matrix, where the rows represent the scores of chosen responses
    # and the columns represent the scores of rejected responses
    MATRIX_SIZE = 3 # the column and row size of the matrix
    acc_matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
    for result in results:
        for i in range(len(result["result"])):
            for j in range(len(result["result"])):
                if result["result"][i] == 1:
                    acc_matrix[i][j] += 1
    
    # compute the accuracy by dividing the number of correct comparisons by the total number of comparisons
    acc_matrix /= len(results)
    # compute the hard,normal,easy accuracy
    # hard accuracy: the average of the upper-right triangle of the matrix
    # namely chosen responses with less fancy style compared to rejected responses with more fancy style
    upper_right_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
    hard_acc = np.sum(np.triu(acc_matrix, 1)) / upper_right_count
    # normal accuracy: the average of the diagonal of the matrix
    # namely chosen responses with the same style compared to rejected responses with the same style
    normal_acc = np.mean(np.diag(acc_matrix))
    # easy accuracy: the average of the lower-left triangle of the matrix
    # namely chosen responses with more fancy style compared to rejected responses with less fancy style
    lower_left_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
    easy_acc = np.sum(np.tril(acc_matrix, -1)) / lower_left_count
    
    return {
        "hard_acc": hard_acc,
        "normal_acc": normal_acc,
        "easy_acc": easy_acc
    }

def compute_accuracy(results: List[Dict[str, Any]]) -> Dict[str, float]:
    if 'domain' in results[0]:
        # this indicates this is total_dataset.json
        print('We are handling total_dataset.json')
        print('Splitting the dataset by domain...')
        # thus we need to split the results into different domains
        split_results = split_dataset_by_domain(results)
        domain_results = {}
        for domain in split_results:
            domain_results[domain] = compute_accuracy(split_results[domain])
        domain_avg_results = {}
        for domain in domain_results:
            domain_avg_results[domain] = np.mean(list(domain_results[domain].values()))
        domain_hard_normal_easy_acc = {
            "hard_acc": np.mean([domain_results[domain]["hard_acc"] for domain in domain_results]),
            "normal_acc": np.mean([domain_results[domain]["normal_acc"] for domain in domain_results]),
            "easy_acc": np.mean([domain_results[domain]["easy_acc"] for domain in domain_results])
        }
        total_avg_acc = np.mean([domain_avg_results[domain] for domain in domain_avg_results])
        # merge the results into one falten dictionary
        final_results = {}
        # merge domain_avg_results into final_results
        final_results.update(domain_avg_results)
        # merge domain_hard_normal_easy_acc into final_results
        final_results.update(domain_hard_normal_easy_acc)
        # merge total_avg_acc into final_results
        final_results.update({"total_avg_acc": total_avg_acc})
        return final_results
            
    
    # results is a list of dictionaries, each dictionary contains the following keys:
    # score_chosen: [float, float, float], the scores of the chosen responses
    # score_rejected: [float, float, float], the scores of the rejected responses
    # the scores are in the order of [concise, detailed_plain, detailed_markdown]
    # we will compare the scores of chosen responses and rejected responses iteratively
    # formatted as a 3x3 matrix, where the rows represent the scores of chosen responses
    # and the columns represent the scores of rejected responses
    MATRIX_SIZE = 3 # the column and row size of the matrix
    acc_matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
    for result in results:
        for i in range(len(result["score_chosen"])):
            for j in range(len(result["score_rejected"])):
                if result["score_chosen"][i] > result["score_rejected"][j]:
                    acc_matrix[i][j] += 1
    
    # compute the accuracy by dividing the number of correct comparisons by the total number of comparisons
    acc_matrix /= len(results)
    # compute the hard,normal,easy accuracy
    # hard accuracy: the average of the upper-right triangle of the matrix
    # namely chosen responses with less fancy style compared to rejected responses with more fancy style
    upper_right_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
    hard_acc = np.sum(np.triu(acc_matrix, 1)) / upper_right_count
    # normal accuracy: the average of the diagonal of the matrix
    # namely chosen responses with the same style compared to rejected responses with the same style
    normal_acc = np.mean(np.diag(acc_matrix))
    # easy accuracy: the average of the lower-left triangle of the matrix
    # namely chosen responses with more fancy style compared to rejected responses with less fancy style
    lower_left_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
    easy_acc = np.sum(np.tril(acc_matrix, -1)) / lower_left_count
    
    return {
        "hard_acc": hard_acc,
        "normal_acc": normal_acc,
        "easy_acc": easy_acc
    }




def load_eval_dataset(
    raw_Dataset: Dataset = None,
    core_set: bool = True,
    custom_dialogue_formatting: bool = False,
    conv: Conversation = None,
    tokenizer: PreTrainedTokenizer = None,
    logger: logging.Logger = None,
    keep_columns: List[str] = ["text_chosen", "text_rejected", "id"],
    return_extra_data: bool = False,
    max_turns: int = None,
) -> tuple[Dataset, list[str]]:
    """
    Loads either the core eval set for HERM or the existing preference data test sets.

    Args:
        core_set: if True, load the core eval set for HERM.
        custom_dialogue_formatting: if True, format the dialogue as needed for custom models (e.g. SHP and PairRM).
        conv: fastchat conversation template.
                If None (default) the passed tokenizer needs to have a usable chat template.
        tokenizer: HuggingFace tokenizer to use. The tokenizer's chat template, if available, has precedence over conv.
        logger: logger to use for logging. If None (default), no logging is done.
        keep_columns: list of columns to keep in the dataset.
        max_turns: maximum number of turns in the dialogue (usually even). If None (default), no filtering is done.

    Returns:
        dataset: loaded dataset with required properties.
        subsets: list of subsets for the corresponding samples in the dataset.
    """
    if raw_Dataset is not None:
        raw_dataset = raw_Dataset
    elif core_set:
        raw_dataset = load_from_disk("data/reward-bench")
        raw_dataset = raw_dataset['filtered']
    else:
        raw_dataset = load_dataset(EXTRA_PREF_SETS)
        modified_datasets = []

        # Iterate over each subset in the DatasetDict
        for subset_name, subdataset in raw_dataset.items():
            # if subset column exists, move to subsubset (for pref sets)
            if "subset" in subdataset.column_names:
                subdataset = subdataset.rename_column("subset", "subsubset")

            # Add a new column 'subset' to the dataset with the subset name
            subdataset = subdataset.add_column("subset", [subset_name] * len(subdataset))

            # Append the modified dataset to the list
            # remove pku_safer and pku_better from the dict, no longer part of the benchmark
            if subset_name not in ["pku_safer", "pku_better"]:
                modified_datasets.append(subdataset)

        # Concatenate all the modified datasets into one dataset
        raw_dataset = concatenate_datasets(modified_datasets)

    # Apply chat template
    if not custom_dialogue_formatting:
        usable_tokenizer = check_tokenizer_chat_template(tokenizer)

        # assert either conv is passed or tokenizer has chat_template
        assert conv is not None or usable_tokenizer

        if usable_tokenizer:
            if logger is not None:
                logger.info("*** Preparing dataset with HF Transformers ***")
            # docs https://huggingface.co/docs/transformers/main/en/chat_templating
            dataset = raw_dataset.map(
                prepare_dialogue_from_tokenizer,
                fn_kwargs={"tokenizer": tokenizer},
                num_proc=8,
                load_from_cache_file=False,
            )

        # else use FastChat to get chat template
        else:
            if logger is not None:
                logger.info("*** Preparing dataset with FastChat ***")
            dataset = raw_dataset.map(
                prepare_dialogue,
                fn_kwargs={"dialogue_template": conv},
                num_proc=8,  # using >1 process causes issues with re-assigning prompt in example
                load_from_cache_file=False,
            )
    else:
        if logger is not None:
            logger.info("*** Preparing dataset with custom formatting ***")

        def map_conversations(example, core_set=True):
            if core_set:
                example["text_chosen"] = [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["chosen"]},
                ]
                example["text_rejected"] = [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["rejected"]},
                ]
            else:
                prompt = example["prompt"]
                example["text_chosen"] = prompt + [{"role": "assistant", "content": example["chosen"]}]
                example["text_rejected"] = prompt + [{"role": "assistant", "content": example["rejected"]}]
            return example

        dataset = raw_dataset.map(
            map_conversations,
            fn_kwargs={"core_set": core_set},
            num_proc=8,
        )

    if max_turns is not None:
        assert max_turns > 0, "max_turns must be greater than 0"

        # filter long answers (MT Bench prompt as 1 or 2 turn examples)
        def filter_long_turns(batch):
            return len(batch["text_chosen"]) <= max_turns

        dataset = dataset.filter(filter_long_turns)

    # take column subset from dataset
    subsets = dataset["subset"]

    # remove columns if set and not custom_dialogue_formatting
    all_cols = dataset.column_names
    dataset = dataset.remove_columns([c for c in all_cols if c not in keep_columns])

    return dataset, subsets



if __name__ == "__main__":
    # test the function
    pass