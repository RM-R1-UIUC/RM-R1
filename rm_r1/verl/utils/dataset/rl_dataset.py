from __future__ import annotations

import verl.utils.torch_functional as verl_F
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers import ProcessorMixin
from verl.utils.model import compute_position_id_with_mask
from copy import deepcopy


class RubricRMDataset(Dataset):
    """
    Rubric RM dataset
    """

    def __init__(
        self,
        files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        processor: ProcessorMixin | None = None,
        prompt_key='prompt',
        image_key='images',
        max_prompt_length=1024,
        return_raw_chat=False,
        truncation='error',
        filter_overlong_prompts=False,
    ):

        self.files = files
        assert files.startswith('gaotang/'), ValueError(
            'hard coded check.',
        )
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length

        self.return_raw_chat = return_raw_chat
        self.truncation = truncation
        self.filter_overlong_prompts = filter_overlong_prompts

        # whether to store the dataset in state_dict()
        # default not store

        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        self.dataset = load_dataset(self.files)['train'].to_dict()
        self.dataset = [{
            self.prompt_key: self.dataset['context_messages'][i],
            'reward_model': {
                'ground_truth': self.dataset['winner'][i],
            },
            'data_source': self.files,
        } for i in range(len(self.dataset['context_messages']))]

        print(f'dataset len: {len(self.dataset)}')

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key

            valid_dataset = []

            for data_point in tqdm(self.dataset, desc='Filtering Dataset'):
                if len(tokenizer.apply_chat_template(data_point[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length:
                    valid_dataset.append(data_point)

            self.dataset = valid_dataset

            print(f'filter dataset len: {len(self.dataset)}')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = deepcopy(self.dataset[index])

        chat = row_dict.pop(self.prompt_key)

        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False,
        )

        is_multi_modal = self.image_key in row_dict
        assert not is_multi_modal
        raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(
            raw_prompt, add_special_tokens=False,
        )

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        # index = row_dict.get("extra_info", {}).get("index", 0)
        # row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['dataset']
        return state
