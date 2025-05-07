import copy, re
from datasets import load_dataset, concatenate_datasets, Dataset
from collections import Counter

import argparse 
from tqdm import tqdm 




SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the Client's question displayed below.\n\n"
    "First, classify the task into one of two categories: <type>Reasoning</type> or <type>Chat</type>.\n"
    "- Use <type>Reasoning</type> for tasks that involve math, coding, or require domain knowledge, multi-step inference, logical deduction, or combining information to reach a conclusion.\n"
    "- Use <type>Chat</type> for tasks that involve open-ended or factual conversation, stylistic rewrites, safety questions, or general helpfulness requests without deep reasoning.\n\n"
    
    "If the task is Reasoning:\n"
    "1. Solve the Client's question yourself and present your final answer within <solution>...</solution> tags.\n"
    "2. Evaluate the two Chatbot responses based on correctness, completeness, and reasoning quality, referencing your own solution.\n"
    "3. Include your evaluation inside <eval>...</eval> tags, quoting or summarizing the Chatbots using the following tags:\n"
    "   - <quote_A>...</quote_A> for direct quotes from Chatbot A\n"
    "   - <summary_A>...</summary_A> for paraphrases of Chatbot A\n"
    "   - <quote_B>...</quote_B> for direct quotes from Chatbot B\n"
    "   - <summary_B>...</summary_B> for paraphrases of Chatbot B\n"
    "4. End with your final judgment in the format: <answer>[[A]]</answer> or <answer>[[B]]</answer>\n\n"

    "If the task is Chat:\n"
    "1. Generate evaluation criteria (rubric) tailored to the Client's question and context, enclosed in <rubric>...</rubric> tags.\n"
    "2. Assign weights to each rubric item based on their relative importance.\n"
    "3. Inside <rubric>, include a <justify>...</justify> section explaining why you chose those rubric criteria and weights.\n"
    "4. Compare both Chatbot responses according to the rubric.\n"
    "5. Provide your evaluation inside <eval>...</eval> tags, using <quote_A>, <summary_A>, <quote_B>, and <summary_B> as described above.\n"
    "6. End with your final judgment in the format: <answer>[[A]]</answer> or <answer>[[B]]</answer>\n\n"

    "Important Notes:\n"
    "- Be objective and base your evaluation only on the content of the responses.\n"
    "- Do not let response order, length, or Chatbot names affect your judgment.\n"
    "- Follow the response format strictly depending on the task type.\n\n"

    "Your output must follow one of the two formats below:\n\n"
    "For Reasoning:\n"
    "<type>Reasoning</type>\n\n"
    "<solution> your own solution for the problem </solution>\n\n"
    "<eval>\n"
    "  include direct comparisons supported by <quote_A>...</quote_A> or <summary_A>...</summary_A>, and <quote_B>...</quote_B>, or <summary_B>...</summary_B>\n"
    "</eval>\n\n"
    "<answer>[[A/B]]</answer>\n\n"

    "For Chat:\n"
    "<type>Chat</type>\n\n"
    "<rubric>\n"
    "  detailed rubric items\n"
    "  <justify> justification for the rubric </justify>\n"
    "</rubric>\n\n"
    "<eval>\n"
    "  include direct comparisons supported by <quote_A>...</quote_A> or <summary_A>...</summary_A>, and <quote_B>...</quote_B>, or <summary_B>...</summary_B> tags\n"
    "</eval>\n\n"
    "<answer>[[A/B]]</answer>"
)

TEMPALTE_SINGLE = [
    {
        'role': 'system',
        'content': SYSTEM_PROMPT
    },
    {
        'role': 'user',
        'content': (
            "[Client Question]\n{question}\n\n[The Start of Chatbot A's Response]\n{answer_a}\n[The End of Chatbot A's Response]\n\n"
            "[The Start of Chatbot B's Response]\n{answer_b}\n[The End of Chatbot B's Response]"
        )
    }
]

TEMPLATE_MULTI = [
    {
        'role': 'system',
        'content': SYSTEM_PROMPT
    },
    {
        'role': 'user',
        'content': (
            "[The Start of the Conversation between Chatbot A and the Client]\n{conversation_1}\n[The End of the Conversation between Chatbot A and the Client]\n\n"
            "[The Start of the Conversation between Chatbot B and the Client]\n{conversation_2}\n[The End of the Conversation between Chatbot B and the Client]"
        )
    }
]


CURRENT_NUM = 0

def convert_sky_data_single(input):
    assert len(input) == 2 and input[0]['role'] == 'user' and input[1]['role'] == "assistant"

    question = input[0]['content']
    answer = input[1]['content']
    return question, answer 

def convert_sky_data_multi(input, assistant_name="Assistant"):
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

 

def get_smaller_dataset_sky():
    global CURRENT_NUM
    ds = load_dataset("Skywork/Skywork-Reward-Preference-80K-v0.2")

    context_messages = []
    winner = []

    num_single, num = 0, 0
    for item in ds['train']:
        if item['source'] != "magpie_ultra":

            input_chosen = item['chosen'] 
            input_rej = item['rejected']

            num += 1
            if (len(input_chosen) == 2 and input_chosen[0]['role'] == 'user' and input_chosen[1]['role'] == "assistant") \
                and (len(input_rej) == 2 and input_rej[0]['role'] == 'user' and input_rej[1]['role'] == "assistant"):
                single = True
                num_single += 1  
            else:
                single = False 

            if single:
                question_chosen, answer_chosen = convert_sky_data_single(item['chosen'])
                question_rej, answer_rej = convert_sky_data_single(item['rejected'])
                assert question_chosen == question_rej

                if CURRENT_NUM % 2 == 0:
                    answer_a = answer_chosen
                    answer_b = answer_rej
                    winner.append('model_a')
                else:
                    answer_a = answer_rej
                    answer_b = answer_chosen
                    winner.append('model_b') 
                
                curr_prompt = copy.deepcopy(TEMPALTE_SINGLE) 
                curr_prompt[1]['content'] = curr_prompt[1]['content'].format(
                    question=question_chosen,
                    answer_a=answer_a,
                    answer_b=answer_b
                )

                context_messages.append(curr_prompt)
                CURRENT_NUM += 1 

            else:
                if CURRENT_NUM % 2 == 0:
                    conversation_1 = convert_sky_data_multi(item['chosen'], assistant_name="Chatbot A")
                    conversation_2 = convert_sky_data_multi(item['rejected'], assistant_name="Chabot B")
                    winner.append('model_a')
                else:
                    conversation_1 = convert_sky_data_multi(item['rejected'], assistant_name="Chatbot A")
                    conversation_2 = convert_sky_data_multi(item['chosen'], assistant_name="Chatbot B")
                    winner.append('model_b') 
                
                curr_prompt = copy.deepcopy(TEMPLATE_MULTI) 
                curr_prompt[1]['content'] = curr_prompt[1]['content'].format(
                    conversation_1=conversation_1,
                    conversation_2=conversation_2,
                )

                context_messages.append(curr_prompt)
                CURRENT_NUM += 1 

    print(f"Num_single {num_single}, total num: {num}")
    dataset = Dataset.from_dict({
        'context_messages': context_messages,
        'winner': winner
    })
    return dataset 


def collect_code_data_sky():
    global CURRENT_NUM 
    dataset = load_dataset("Vezora/Code-Preference-Pairs", split="train")  # 'train' split as an example
    shuffled = dataset.shuffle(seed=42)
    subset_15k = shuffled.select(range(8000))
    remainder = shuffled.select(range(8000, len(shuffled)))

    context_messages = []
    winner = []

    for item in subset_15k:
        question = item['input']
        answer_chosen = item['accepted']
        answer_rej = item['rejected']

        if CURRENT_NUM % 2 == 0:
            answer_a = answer_chosen
            answer_b = answer_rej
            winner.append('model_a')
        else:
            answer_a = answer_rej
            answer_b = answer_chosen
            winner.append('model_b')
        

        curr_prompt = copy.deepcopy(TEMPALTE_SINGLE) 
        curr_prompt[1]['content'] = curr_prompt[1]['content'].format(
            question=question,
            answer_a=answer_a,
            answer_b=answer_b
        )

        context_messages.append(curr_prompt)
        CURRENT_NUM += 1 
    
    dataset = Dataset.from_dict({
        'context_messages': context_messages,
        'winner': winner
    })

    return dataset, remainder

def collect_math_dpo_10k():
    global CURRENT_NUM 
    dataset = load_dataset("xinlai/Math-Step-DPO-10K", split="train")  # 'train' split as an example
    context_messages = []
    winner = []

    for item in dataset:
        question = item['prompt']
        prefix = item['initial_reason_steps']
        chosen = item['full_chosen']
        rejected = item['full_rejected']

        answer_chosen = prefix + chosen 
        answer_rej = prefix + rejected 

        if CURRENT_NUM % 2 == 0:
            answer_a = answer_chosen
            answer_b = answer_rej
            winner.append('model_a')
        else:
            answer_a = answer_rej
            answer_b = answer_chosen
            winner.append('model_b')

        curr_prompt = copy.deepcopy(TEMPALTE_SINGLE) 
        curr_prompt[1]['content'] = curr_prompt[1]['content'].format(
            question=question,
            answer_a=answer_a,
            answer_b=answer_b
        )

        context_messages.append(curr_prompt)
        CURRENT_NUM += 1
    
    dataset = Dataset.from_dict({
        'context_messages': context_messages,
        'winner': winner
    })

    return dataset


def collect_dataset():
    math_data = collect_math_dpo_10k()
    code_data, _ = collect_code_data_sky()
    sky_data = get_smaller_dataset_sky()
    code_sky = concatenate_datasets([code_data, sky_data, math_data])

    print(Counter(code_sky["winner"])) # Show the respective class number
    code_sky.push_to_hub("your_preference_dataset")

if __name__ == '__main__':
    collect_dataset()    
    
