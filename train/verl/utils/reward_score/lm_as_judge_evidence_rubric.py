# REWARD_PATH=./rubric_rm/verl/utils/reward_score/lm_as_judge_evidence_rubric_reasoning.py

def answer_reward(solution_str: str, answer: str) ->float:
    pred = solution_str[-80:]

    if answer == 'model_a':
        if '<answer>[[A]]</answer>' in pred and '<answer>[[B]]</answer>' not in pred:
            return 1.0
        else:
            return -1.0
    elif answer == 'model_b':
        if '<answer>[[B]]</answer>' in pred and '<answer>[[A]]</answer>' not in pred:
            return 1.0
        else:
            return -1.0
    else:
        raise NotImplementedError('Check your dataset label!')

def lm_as_judge_match(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
):
    r = answer_reward(solution_str, ground_truth)
    
    return r 