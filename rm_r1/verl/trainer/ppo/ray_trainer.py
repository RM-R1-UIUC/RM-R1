from __future__ import annotations

import torch
from omegaconf import OmegaConf
from omegaconf import open_dict
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.trainer.ppo.ray_trainer import RayPPOTrainer as _RayPPOTrainer
from verl.utils.dataset.rl_dataset import collate_fn

from rubric_rm.verl.utils.dataset.rl_dataset import RubricRMDataset


class RubricRMRayPPOTrainer(_RayPPOTrainer):
    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RubricRMDataset(
            files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            image_key=self.config.data.get('image_key', 'images'),
            max_prompt_length=self.config.data.max_prompt_length,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation=self.config.data.truncation,
            filter_overlong_prompts=self.config.data.filter_overlong_prompts,
        )
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(
                self.config.data.get('seed', 1),
            )
            sampler = RandomSampler(
                data_source=self.train_dataset, generator=train_dataloader_generator,
            )
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            num_workers=8,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        self.val_dataset = RubricRMDataset(
            files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            image_key=self.config.data.get('image_key', 'images'),
            max_prompt_length=self.config.data.max_prompt_length,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation=self.config.data.truncation,
            filter_overlong_prompts=self.config.data.filter_overlong_prompts,
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader,
        ) == 1, 'Validation dataloader must have a single batch, which inference engines will schedule the memory themselves.'

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(
            self.train_dataloader,
        ) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps
