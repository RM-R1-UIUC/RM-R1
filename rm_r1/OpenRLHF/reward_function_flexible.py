# reward_func.py
import torch

import re

# Explanation of the core snippet, repeated for each tag:
# <rubric>\s*(\S.*?\S|\S)\s*</rubric>
#
# - \s* allows leading whitespace inside the tag before the real content
# - (\S.*?\S|\S) ensures there's at least one non-whitespace character:
#     - \S.*?\S means there's a non-whitespace at the start and at the end (with anything in-between).
#     - |\S covers the case of a single-character non-whitespace (so it doesn't force two).
# - \s* then allows trailing whitespace (e.g. <rubric>   content   </rubric>)
#
# We wrap these in  .*?  (non-greedy, DOTALL) to allow anything around/between the tags.

pattern = re.compile(
    r'^'
    r'.*?<rubric>\s*(\S.*?\S|\S)\s*</rubric>'
    r'.*?<eval>\s*(\S.*?\S|\S)\s*</eval>'
    r'.*?<answer>\s*(\S.*?\S|\S)\s*</answer>'
    r'.*?$',
    re.DOTALL
)

def validate_string_format(s: str) -> bool:
    """
    Returns True if 's' contains (in order):
      - <rubric> with at least one non-whitespace char
      - <eval> with at least one non-whitespace char
      - <answer> with at least one non-whitespace char
    and potentially anything before/between/after those blocks.
    Otherwise returns False.
    """
    return bool(pattern.match(s))


def reward_func(queries, prompts, labels):
    r = []
    for answer, pred in zip(labels, queries):
        
        if validate_string_format(pred):
            pred = pred.split("</eval>")[-1].strip()
            if answer == 'model_a':
                if '[[A]]' in pred and '[[B]]' not in pred:
                    r.append(torch.tensor([1.0])) 
                else:
                    r.append(torch.tensor([-1.0]))    
            elif answer == 'model_b':
                if '[[B]]' in pred and '[[A]]' not in pred:
                    r.append(torch.tensor([1.0]))
                else:
                    r.append(torch.tensor([-1.0]))
            else:
                raise NotImplementedError("Check your dataset label!")
        else:
            r.append(torch.tensor([-1.0])) 
    r = torch.cat(r)
    return r

# def reward_func(queries, prompts, labels):
#     preds = [x.split("</eval>")[-1].strip() if '</eval>' in x else 'error' for x in queries]
#     r = []
#     for answer, pred in zip(labels, preds):
#         if answer == 'model_a':
#             if '[[A]]' in pred and '[[B]]' not in pred:
#                 r.append(torch.tensor([1.0])) 
#             else:
#                 r.append(torch.tensor([-1.0]))    
#         elif answer == 'model_b':
#             if '[[B]]' in pred and '[[A]]' not in pred:
#                 r.append(torch.tensor([1.0]))
#             else:
#                 r.append(torch.tensor([-1.0]))
#         else:
#             raise NotImplementedError("Check your dataset label!")
#     r = torch.cat(r)
#     return r