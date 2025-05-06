# reward_func.py
import torch

def reward_func(queries, prompts, labels):
    preds = [x.split("</eval>")[-1].strip() if '</eval>' in x else 'error' for x in queries]
    r = []
    for answer, pred in zip(labels, preds):
        
        # print("answer: ", answer) 
        # print("predict: ", pred)
        # print("pred == <answer>[[A]]</answer>: ", '<answer>[[A]]</answer>' == pred)
        # print("pred == '<answer>[[B]]</answer>': ", '<answer>[[B]]</answer>' == pred)
        if answer == 'model_a':
            if '<answer>[[A]]</answer>' in pred[-40:] and len(pred) <= 45 and '[[B]]' not in pred:
                r.append(torch.tensor([1.0])) 
            else:
                r.append(torch.tensor([-1.0]))

            # if '[[A]]' in pred and '[[B]]' not in pred:
            #     r.append(torch.tensor([1.0])) 
            # else:
            #     r.append(torch.tensor([-1.0]))    
            # if '[[A]]' in pred:
                # r.append(torch.tensor([1.0]))
            # else:
                # r.append(torch.tensor([-1.0]))
        elif answer == 'model_b':
            if '<answer>[[B]]</answer>' in pred[-40:] and len(pred) <=45 and '[[A]]' not in pred:
                # There are [EOS] prevents <answer>[[A]]</answer><answer>[[B]]</answer>
                r.append(torch.tensor([1.0]))
            else:
                r.append(torch.tensor([-1.0]))

            # if '[[B]]' in pred and '[[A]]' not in pred:
            #     r.append(torch.tensor([1.0]))
            # else:
            #     r.append(torch.tensor([-1.0]))
            # if '[[B]]' in pred:
                # r.append(torch.tensor([1.0]))
            # elif '[[B]]'
            # else:
                # r.append(torch.tensor([-1.0]))
        else:
            raise NotImplementedError("Check your dataset label!")
    r = torch.cat(r)
    return r