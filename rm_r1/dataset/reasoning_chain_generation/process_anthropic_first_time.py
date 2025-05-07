import json 

dir = "absolute_path_to/corresponding_input_ds.jsonl"
with open(dir, "r") as f:
    data = json.load(f) 

data_dict = {item['custom_id'] : {"context_messages": item['context_messages'], "winner": item['winner']} for item in data}
with open("absolute_path_to/batch_results.jsonl", 'r') as json_file:
    json_list = list(json_file)
results = [json.loads(json_str) for json_str in json_list]

META_RESULT = []

for item in results:
    id = item['custom_id']
    curr_result = {}
    curr_result['custom_id'] = id
    curr_result['context_messages'] = data_dict[id]['context_messages']
    curr_result['winner'] = data_dict[id]['winner']
    curr_result['sft_response'] = item['result']['message']['content'][0]['text']
    META_RESULT.append(curr_result)

def get_num_correct():
    total, correct = 0, 0 
    for item in META_RESULT:
        total += 1
        winner = item['winner'] 
        pred = item['sft_response'][-100:]

        if winner == "model_a":
            if '[[A]]' in pred:
                correct += 1 
        elif winner == "model_b":
            if '[[B]]' in pred:
                correct += 1
    return correct, total 
c, t = get_num_correct()
print("Correct: ", c)
print("Total: ", t) # 3/4 correct from our usage

from copy import  deepcopy 

RESULT = []

for item in META_RESULT:
    curr_item = deepcopy(item)
    winner = item['winner'] 
    pred = item['sft_response'][-100:]

    correct = False 
    if winner == "model_a":
        if '[[A]]' in pred:
            correct = True 
    elif winner == "model_b":
        if '[[B]]' in pred:
            correct = True
    if correct:
        curr_item['correct'] = True 
    else:
        curr_item['correct'] = False 
    RESULT.append(curr_item) 

correct_result = [item for item in RESULT if item['correct'] == True]
incorrect_result = [item for item in RESULT if item['correct'] == False]

with open("abs_path_to/claude_results_correct_first_time.jsonl", 'w') as json_file:
    json.dump(correct_result, json_file, indent=4)
with open("abs_path_to/claude_results_incorrect_first_time.jsonl", 'w') as json_file:
    json.dump(incorrect_result, json_file, indent=4)