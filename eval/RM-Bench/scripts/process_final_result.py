import argparse 
import json 
import os 
import numpy as np 
from typing import List, Dict, Any
from pathlib import Path 

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True) 
parser.add_argument("--model_save_name", type=str, required=True)
parser.add_argument(
    "--meta_result_save_dir", default=None, type=str
)
args = parser.parse_args()


model_name = args.model.split("/")[-1]
ref_model_name = "REWORD_MODEL"
if args.meta_result_save_dir is None: 
    output_dir = f"results/Gen_RMs/{args.model_save_name}"
else:
    output_dir = f"{args.meta_result_save_dir}/{args.model_save_name}/RM-Bench/logs"

output_path1 = os.path.join(output_dir, f"total_dataset_1_{model_name}.json") 
output_path2 = os.path.join(output_dir, f"total_dataset_2_{model_name}.json") 
output_path3 = os.path.join(output_dir, f"total_dataset_3_{model_name}.json") 

with open(output_path1) as json_file:
    data1 = json.load(json_file)

with open(output_path3) as json_file:
    data2 = json.load(json_file)

with open(output_path3) as json_file:
    data3 = json.load(json_file)




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
        # for i in range(len(result["result"])):
        #     for j in range(len(result["result"])):
        #         if result["result"][i] == 1:
        acc_matrix += result['acc_matrix']
    
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

example1, example2, example3 = data1[0], data2[0], data3[0] 
res1, res2, res3 = example1['result'], example2['result'], example3['result']

"""
[chosen 0 - rejected 0, chosen 1 - rejected 1, chosen 2 - rejected 2] res 1
[chosen 0 - rejected 1, chosen 1 - rejected 2, chosen 2 - rejected 0] res 2
[chosen 0 - rejected 2, chosen 1 - rejected 0, chosen 2 - rejected 1] res 3

->>>

[chosen 0 - rejected 0, chosen 0 - rejected 1, chosen 0 - rejected 2,
chosen 1 - rejected 0, chosen 1 - rejected 1, chosen 1 - rejected 2,
chosen 2 - rejected 0, chosen 2 - rejected 1, chosen 2 - rejected 2]

"""


# MATRIX_SIZE = 3 # the column and row size of the matrix
# acc_matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
# acc_matrix[0, 0] = res1[0] 
# acc_matrix[0, 1] = res2[0]
# acc_matrix[0, 2] = res3[0]

# acc_matrix[1, 0] = res3[1]
# acc_matrix[1, 1] = res1[1]
# acc_matrix[1, 2] = res2[1]

# acc_matrix[2, 0] = res2[2]
# acc_matrix[2, 1] = res3[2]
# acc_matrix[2, 2] = res1[2] 

meta_data = [] 
for example1, example2, example3 in zip(data1, data2, data3):
    assert example1["id"] == example2["id"] == example3["id"]
    res1, res2, res3 = example1['result'], example2['result'], example3['result']
    
    MATRIX_SIZE = 3 # the column and row size of the matrix
    acc_matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
    acc_matrix[0, 0] = res1[0] 
    acc_matrix[0, 1] = res2[0]
    acc_matrix[0, 2] = res3[0]

    acc_matrix[1, 0] = res3[1]
    acc_matrix[1, 1] = res1[1]
    acc_matrix[1, 2] = res2[1]

    acc_matrix[2, 0] = res2[2]
    acc_matrix[2, 1] = res3[2]
    acc_matrix[2, 2] = res1[2] 

    # print(acc_matrix)
    item = {
        "id": example1["id"],
        "prompt": example1['prompt'],
        "domain": example1['domain'],
        "acc_matrix": acc_matrix
    }
    meta_data.append(item)

final_res = compute_accuracy_gen(meta_data)
print(final_res)

with open(f"{output_dir}/final_result.json", "w") as f:
    json.dump(final_res, f, indent=4)


if args.meta_result_save_dir is not None:
    meta_result_save_dir = Path(args.meta_result_save_dir) / args.model_save_name / "RM-Bench"
    meta_result_save_dir.mkdir(exist_ok=True, parents=True)

    with open(meta_result_save_dir / "final_score.json", 'w') as f:
        json.dump(final_res, f, indent=4)