import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel


class LlamaGenerativeRewardModel(PreTrainedModel):
    config_class = AutoConfig
    _no_split_modules = []

    def __init__(self, config):
        super().__init__(config)
        model = AutoModelForCausalLM.from_config(config)
        self.model = model

    def forward(self, 
                *args,
                **kwargs):
        outputs = self.model(*args, **kwargs)
        return outputs.logits[:, -1]


class GenerativeRMPipeline:
    
    def __init__(self, task, model, tokenizer, template=''):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, samples, return_inputs=False, **kwargs):
        _ = kwargs.get("batch_size", 1)
        truncation = kwargs.get("truncation", True)
        padding = kwargs.get("padding", True)
        max_length = kwargs.get("max_length", 2048)

        import ipdb; ipdb.set_trace()

        inputs = self.tokenizer(
            samples,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            # return_special_tokens_mask=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        import ipdb; ipdb.set_trace()

        return outputs.logits

