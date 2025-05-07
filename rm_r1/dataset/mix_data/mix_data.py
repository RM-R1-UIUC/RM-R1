import copy, re
from datasets import load_dataset, concatenate_datasets, Dataset
from collections import Counter

import argparse 
from tqdm import tqdm 
def parse_embedded_mpt_blocks(content: str, default_role: str = "assistant"):
    """
    Parse a single message 'content' string that may contain embedded
    <|im_start|>role\n ... <|im_end|> blocks (MPT-style).
    
    Returns a list of { "role": ..., "content": ... }.
    
    - Any text before the first <|im_start|> is assigned to `default_role`.
    - Each <|im_start|>block...<|im_end|> is split out as its own message,
      using the line immediately after <|im_start|> as the role.
    - If a given block has empty content, it's skipped.
    """
    # If no <|im_start|> in the string, nothing to expand—treat as a single block
    if "<|im_start|>" not in content:
        return [{"role": default_role, "content": content.strip()}]
    
    pieces = []
    
    # 1) Grab any leading text before the first <|im_start|>
    first_idx = content.find("<|im_start|>")
    leading = content[:first_idx].strip()
    if leading:
        pieces.append({"role": default_role, "content": leading})
    
    pattern = r"<\|im_start\|>([^\n]+)\n(.*?)<\|im_end\|>"
    blocks = re.findall(pattern, content, flags=re.DOTALL)
    
    for raw_role, block_content in blocks:
        role = raw_role.strip()
        text = block_content.strip()
        # if text:  # skip empty blocks
            # pieces.append({"role": role, "content": text})
        pieces.append({"role": role, "content": text})
    
    return pieces


def convert_to_multiturn(messages):
    """
    Given a list of dicts like:
       [{"role": "user", "content": "..."},
        {"role": "assistant", "content": "... possibly with <|im_start|> ... <|im_end|> ..."}]
    return a new list with multi-turn expansions extracted.
    """
    new_messages = []
    
    for msg in messages:
        if msg["role"] != "assistant":
            # Keep non-assistant messages as-is
            new_messages.append(msg)
        else:
            # Parse out any embedded MPT blocks from assistant messages
            expanded = parse_embedded_mpt_blocks(msg["content"], default_role="assistant")
            new_messages.extend(expanded)
    
    return new_messages



# "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. 
# You should choose the assistant that follows the user's instructions and answers the user's question better. Y
# our evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, 
# and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. 
# Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. 
# Do not allow the length of the responses to influence your evaluation. 
# Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, 
# output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant 
# B is better, and \"[[C]]\" for a tie.", 
# "prompt_template": "[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]", "description": "Prompt for general questions", "category": "general", "output_format": "[[A]]""


prompt_template_rl_detailed_evidence_single = [
    {
        'role': 'system',
        'content': (
            "Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots "
            "to the Client question displayed below. "
            "Begin your evaluation by first generating the rubric items. Enclose this section within <rubric> and </rubric> tags. "
            "Then, compare the following two conversations between the Client and the AI Chatbots, and provide your evaluation according to the rubric items. "
            "Ensure that the order in which the responses are presented does not influence your decision, and do not "
            "let response length or Chatbot names affect your evaluation. Be as objective as possible. "
            "Additionally, base your evaluation and rubric on the specific text of the provided model responses, and justify your evaluation (and/or rubric) by quoting or summarizing relevant parts of each response. "
            "Whenever you quote Chatbot A verbatim, enclose the quoted text in <quote_A>...</quote_A>. "
            "Whenever you summarize or paraphrase Chatbot A, enclose that summary in <summary_A>...</summary_A>. "
            "Likewise, use <quote_B>...</quote_B> to quote Chatbot B verbatim, and <summary_B>...</summary_B> to summarize or paraphrase Chatbot B. "
            "Enclose your complete evaluation explanation within <eval> and </eval> tags. "
            "After providing your explanation, "
            "output your final verdict by strictly following this format: '<answer>[[A]]</answer>' if Chatbot A is better, '<answer>[[B]]</answer>' if Chatbot B is better."
            "i.e., <rubric>rubric items here</rubric>\n\n<eval>detailed evaluation here, referencing each Chatbot's response using <quote_A>...</quote_A>, <quote_B>...</quote_B>, <summary_A>...</summary_A>, or <summary_B>...</summary_B> as needed</eval>\n\n<answer>[[A/B]]</answer>"
        )
    },
    {
        'role': 'user',
        'content': (
            "[Client Question]\n{question}\n\n[The Start of Chatbot A's Response]\n{answer_a}\n[The End of Chatbot A's Response]\n\n"
            "[The Start of Chatbot B's Response]\n{answer_b}\n[The End of Chatbot B's Response]"
        )
    }
]

prompt_template_rl_detailed_evidence_multi = [
    {
        'role': 'system',
        'content': (
            "Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots "
            "to the Client question displayed below. "
            "Begin your evaluation by first generating the rubric items. Enclose this section within <rubric> and </rubric> tags. "
            "Then, compare the following two conversations between the Client and the AI Chatbots, and provide your evaluation according to the rubric items. "
            "Ensure that the order in which the responses are presented does not influence your decision, and do not "
            "let response length or Chatbot names affect your evaluation. Be as objective as possible. "
            "Additionally, base your evaluation and rubric on the specific text of the provided model responses, and justify your evaluation (and/or rubric) by quoting or summarizing relevant parts of each response. "
            "Whenever you quote Chatbot A verbatim, enclose the quoted text in <quote_A>...</quote_A>. "
            "Whenever you summarize or paraphrase Chatbot A, enclose that summary in <summary_A>...</summary_A>. "
            "Likewise, use <quote_B>...</quote_B> to quote Chatbot B verbatim, and <summary_B>...</summary_B> to summarize or paraphrase Chatbot B. "
            "Enclose your complete evaluation explanation within <eval> and </eval> tags. "
            "After providing your explanation, "
            "output your final verdict by strictly following this format: '<answer>[[A]]</answer>' if Chatbot A is better, '<answer>[[B]]</answer>' if Chatbot B is better."
            "i.e., <rubric>rubric items here</rubric>\n\n<eval>detailed evaluation here, referencing each Chatbot's response using <quote_A>...</quote_A>, <quote_B>...</quote_B>, <summary_A>...</summary_A>, or <summary_B>...</summary_B> as needed</eval>\n\n<answer>[[A/B]]</answer>"
        )
    },
    {
        'role': 'user',
        'content': (
            "[The Start of the Conversation between Chatbot A and the Client]\n{conversation_1}\n[The End of the Conversation between Chatbot A and the Client]\n\n"
            "[The Start of the Conversation between Chatbot B and the Client]\n{conversation_2}\n[The End of the Conversation between Chatbot B and the Client]"
        )
    }
]


prompt_template_rl_detailed_evidence_single_justify_rubric = [
    {
        'role': 'system',
        'content': (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the Client's question or conversation displayed below.\n"
    "Begin your evaluation by first generating the rubric items, and enclose them within <rubric> and </rubric> tags. " 
    "Inside the <rubric> section, also include <justify>...</justify> to explain why you chose those rubric criteria. "
    "Ensure your generated rubric is specific, clear, and tailored to compare the Chatbot responses in the context of the Client's question or conversation.\n"
    "Then, compare the following two conversations between the Client and the AI Chatbots, and provide your evaluation according to the rubric items. " 
    "Base your evaluation on the specific text of each Chatbot's response, and justify your evaluation by quoting or summarizing relevant parts of each response. " 
    "Whenever you quote Chatbot A verbatim, enclose that text in <quote_A>...</quote_A>. " 
    "Whenever you summarize or paraphrase Chatbot A, use <summary_A>...</summary_A>. " 
    "Likewise, for Chatbot B, use <quote_B>...</quote_B> or <summary_B>...</summary_B>. "
    "Enclose your complete evaluation explanation within <eval> and </eval> tags. " 
    "Ensure that the order in which the responses are presented does not influence your decision, and do not let response length or Chatbot names affect your evaluation. Be as objective as possible.\n"
    "After providing your explanation, output your final verdict by strictly following this format: "
    "'<answer>[[A]]</answer>' if Chatbot A is better, or '<answer>[[B]]</answer>' if Chatbot B is better.\n"
    "i.e., <rubric> rubric items here <justify> justification for rubrics </justify> </rubric>\n\n<eval> detailed evaluation here, referencing each Chatbot's response using <quote_A>...</quote_A>, or <summary_A>...</summary_A>, and <quote_B>...</quote_B> or <summary_B>...</summary_B> as needed </eval>\n\n<answer>[[A/B]]</answer>"
)
    },
    {
        'role': 'user',
        'content': (
            "[Client Question]\n{question}\n\n[The Start of Chatbot A's Response]\n{answer_a}\n[The End of Chatbot A's Response]\n\n"
            "[The Start of Chatbot B's Response]\n{answer_b}\n[The End of Chatbot B's Response]"
        )
    }
]

prompt_template_rl_detailed_evidence_single_justify_rubric_Multi = [
    {
        'role': 'system',
        'content': (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the Client's question or conversation displayed below.\n"
    "Begin your evaluation by first generating the rubric items, and enclose them within <rubric> and </rubric> tags. " 
    "Inside the <rubric> section, also include <justify>...</justify> to explain why you chose those rubric criteria. "
    "Ensure your generated rubric is specific, clear, and tailored to compare the Chatbot responses in the context of the Client's question or conversation.\n"
    "Then, compare the following two conversations between the Client and the AI Chatbots, and provide your evaluation according to the rubric items. " 
    "Base your evaluation on the specific text of each Chatbot's response, and justify your evaluation by quoting or summarizing relevant parts of each response. " 
    "Whenever you quote Chatbot A verbatim, enclose that text in <quote_A>...</quote_A>. " 
    "Whenever you summarize or paraphrase Chatbot A, use <summary_A>...</summary_A>. " 
    "Likewise, for Chatbot B, use <quote_B>...</quote_B> or <summary_B>...</summary_B>. "
    "Enclose your complete evaluation explanation within <eval> and </eval> tags. " 
    "Ensure that the order in which the responses are presented does not influence your decision, and do not let response length or Chatbot names affect your evaluation. Be as objective as possible.\n"
    "After providing your explanation, output your final verdict by strictly following this format: "
    "'<answer>[[A]]</answer>' if Chatbot A is better, or '<answer>[[B]]</answer>' if Chatbot B is better.\n"
    "i.e., <rubric> rubric items here <justify> justification for rubrics </justify> </rubric>\n\n<eval> detailed evaluation here, referencing each Chatbot's response using <quote_A>...</quote_A>, or <summary_A>...</summary_A>, and <quote_B>...</quote_B> or <summary_B>...</summary_B> as needed </eval>\n\n<answer>[[A/B]]</answer>"
)
    },
    {
        'role': 'user',
        'content': (
            "[The Start of the Conversation between Chatbot A and the Client]\n{conversation_1}\n[The End of the Conversation between Chatbot A and the Client]\n\n"
            "[The Start of the Conversation between Chatbot B and the Client]\n{conversation_2}\n[The End of the Conversation between Chatbot B and the Client]"
        )
    }
]


GPT_REVISED_SYSTEM_CLASSIFY_PROMPT_EVIDENCE = (
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
    "2. Inside <rubric>, include a <justify>...</justify> section explaining why these criteria are appropriate.\n"
    "3. Compare both Chatbot responses according to the rubric.\n"
    "4. Provide your evaluation inside <eval>...</eval> tags, using <quote_A>, <summary_A>, <quote_B>, and <summary_B> as described above.\n"
    "5. End with your final judgment in the format: <answer>[[A]]</answer> or <answer>[[B]]</answer>\n\n"

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

PROMPT_TEMPLATE_RUBRIC_EVIDENCE_CLASSIFICATION_MULTI = [
    {
        'role': 'system',
        'content': GPT_REVISED_SYSTEM_CLASSIFY_PROMPT_EVIDENCE
    },
    {
        'role': 'user',
        'content': (
            "[The Start of the Conversation between Chatbot A and the Client]\n{conversation_1}\n[The End of the Conversation between Chatbot A and the Client]\n\n"
            "[The Start of the Conversation between Chatbot B and the Client]\n{conversation_2}\n[The End of the Conversation between Chatbot B and the Client]"
        )
    }
]


PROMPT_TEMPLATE_RUBRIC_EVIDENCE_CLASSIFICATION_SINGLE = [
    {
        'role': 'system',
        'content': GPT_REVISED_SYSTEM_CLASSIFY_PROMPT_EVIDENCE
    },
    {
        'role': 'user',
        'content': (
            "[Client Question]\n{question}\n\n[The Start of Chatbot A's Response]\n{answer_a}\n[The End of Chatbot A's Response]\n\n"
            "[The Start of Chatbot B's Response]\n{answer_b}\n[The End of Chatbot B's Response]"
        )
    }
]


GPT_REVISED_SYSTEM_CLASSIFY_WEIGHT_PROMPT_EVIDENCE = (
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

SFT_SINGLE = [
    {
        'role': 'system',
        'content': 
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the Client's question displayed below.\n"
        "You should choose the chatbot that follows the client's instructions and answers the client's question better. " 
        "Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the chatbots. Be as objective as possible. " 
        "Output your final verdict directly by strictly following this format: \"A\" if Chatbot A is better, \"B\" if Chatbot B is better."
    },
    {
        'role': 'user',
        'content': (
            "[Client Question]\n{question}\n\n[The Start of Chatbot A's Response]\n{answer_a}\n[The End of Chatbot A's Response]\n\n"
            "[The Start of Chatbot B's Response]\n{answer_b}\n[The End of Chatbot B's Response]"
        )
    }
]

SFT_MULTI = [
    {
        'role': 'system',
        'content': 
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the Client's question displayed below.\n"
        "You should choose the chatbot that follows the client's instructions and answers the client's question better. " 
        "Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the chatbots. Be as objective as possible. " 
        "Output your final verdict directly by strictly following this format: \"A\" if Chatbot A is better, \"B\" if Chatbot B is better."
    },
    {
        'role': 'user',
        'content': (
            "[The Start of the Conversation between Chatbot A and the Client]\n{conversation_1}\n[The End of the Conversation between Chatbot A and the Client]\n\n"
            "[The Start of the Conversation between Chatbot B and the Client]\n{conversation_2}\n[The End of the Conversation between Chatbot B and the Client]"
        )
    }
]

SYSTEM_ABLATION_NO_RUBRIC = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the Client's question displayed below.\n"
    "You should choose the chatbot that follows the client's instructions and answers the client's question better. " 
    "Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the chatbots. Be as objective as possible. "
    "First, compare the chatbot responses and provide your evaluations. "
    "Then, conclude with your verdict using exactly this format: <answer>[[A]]</answer> if Chatbot A is better, <answer>[[B]]</answer> if Chatbot B is better."
)


SYSTEM_ABLATION_NO_CLASSIFY = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the Client's question displayed below.\n\n"
    "Instructions:\n\n"
    "1. Begin your evaluation by first generating the crtieria (rubric) tailored to the Client's quesiton and context, enclosed in <rubric>...</rubric> tags.\n"
    "2. Assign weights to each rubric item based on their relative importance."
    "3. Inside <rubric>, include a <justify>...</justify> section explaining why you chose those rubric criteria and weights.\n"
    "4. Compare both Chatbot responses according to the rubric.\n"
    "5. Include your evaluation inside <eval>...</eval> tags, quoting or summarizing the Chatbots using the following tags:\n"
    "   - <quote_A>...</quote_A> for direct quotes from Chatbot A\n"
    "   - <summary_A>...</summary_A> for paraphrases of Chatbot A\n"
    "   - <quote_B>...</quote_B> for direct quotes from Chatbot B\n"
    "   - <summary_B>...</summary_B> for paraphrases of Chatbot B\n"
    "6. End with your final judgment in the format: <answer>[[A]]</answer> or <answer>[[B]]</answer>\n\n"

    "Important Notes:\n"
    "- Be objective and base your evaluation only on the content of the responses.\n"
    "- Do not let response order, length, or Chatbot names affect your judgment.\n"

    "Your final output must follow the format below:\n\n"
    "<rubric>\n"
    "  detailed rubric items\n"
    "  <justify> justification for the rubric </justify>\n"
    "</rubric>\n\n"
    "<eval>\n"
    "  include direct comparisons supported by <quote_A>...</quote_A> or <summary_A>...</summary_A>, and <quote_B>...</quote_B>, or <summary_B>...</summary_B> tags\n"
    "</eval>\n\n"
    "<answer>[[A/B]]</answer>"
)


file_path = "/shared/nas2/xiusic/Rubric-RM/document_guideline/model_spec_chat.txt"

with open(file_path, "r", encoding="utf-8") as file:
    guideline_document = file.read()




prompt_template_guideline_single = [
    {
        'role': 'system',
        'content': "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants. "
            "Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. "
            "Enclose your detailed evaluation in <eval> and </eval>. "
            "After providing your explanation, "
            "output your final verdict by strictly following this format: '[[A]]' if Assistant A is better, '[[B]]' if Assistant B is better."
            "i.e., <eval>detailed evaluation here</eval>\n\n<answer>[[A/B]]</answer>.\n"
            "Use evidence from the assistant's responses to guide your evaluation. You can apply different strategies based on the type of question:\n"
            "- For reasoning questions (math, coding), your evaluation should focus on correctness, intermediate steps, key details, and overall helpfulness.\n"
            f"- For non-reasoning questions (chat, safety), your evaluation should follow the guidelines below:\n\n{guideline_document}",
    },
    {
        'role': 'user',
        'content': "[User Question]\n{question}\n\n[The Start of Assistant A's Response]\n{answer_a}\n[The End of Assistant A's Response]\n\n"
        "[The Start of Assistant B's Response]\n{answer_b}\n[The End of Assistant B's Response]"
    }
]

prompt_template_guideline_multi = [
    {
        'role': 'system',
        'content': "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants. "
            "Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. "
            "Enclose your detailed evaluation in <eval> and </eval>. "
            "After providing your explanation, "
            "output your final verdict by strictly following this format: '[[A]]' if Assistant A is better, '[[B]]' if Assistant B is better."
            "i.e., <eval>detailed evaluation here</eval>\n\n<answer>[[A/B]]</answer>.\n"
            "Use evidence from the assistant's responses to guide your evaluation. You can apply different strategies based on the type of question:\n"
            "- For reasoning questions (math, coding), your evaluation should focus on correctness, intermediate steps, key details, and overall helpfulness.\n"
            f"- For non-reasoning questions (chat, safety), your evaluation should follow the guidelines below:\n\n{guideline_document}",
    },
    {
        'role': 'user',
        'content': (
            "[The Start of the Conversation between Assistant A and the User]\n{conversation_1}\n[The End of the Conversation between Assistant A and the User]\n\n"
            "[The Start of the Conversation between Assistant B and the User]\n{conversation_2}\n[The End of the Conversation between Assistant B and the User]"
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
                
                curr_prompt = copy.deepcopy(PROMPT_TEMPLATE_RUBRIC_EVIDENCE_CLASSIFICATION_SINGLE) 
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
                
                curr_prompt = copy.deepcopy(PROMPT_TEMPLATE_RUBRIC_EVIDENCE_CLASSIFICATION_MULTI) 
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
        

        curr_prompt = copy.deepcopy(PROMPT_TEMPLATE_RUBRIC_EVIDENCE_CLASSIFICATION_SINGLE) 
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

        curr_prompt = copy.deepcopy(PROMPT_TEMPLATE_RUBRIC_EVIDENCE_CLASSIFICATION_SINGLE) 
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
    
