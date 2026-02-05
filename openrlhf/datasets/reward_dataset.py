from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils import exist_and_not_none, zero_pad_sequences

"""
Non-pairwise input
"""
def preprocess_nonpair_data(
    data,
    input_template=None,
    prompt_key="prompt",
    response_key="response",
    apply_chat_template=None,
    is_dpo=False,
) -> str:
    # print(apply_chat_template, data, "\n\n", response_key, prompt_key)
    if apply_chat_template:
        if prompt_key:
            prompt = apply_chat_template(data[prompt_key], tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(data[prompt_key] + data[response_key], tokenize=False)[len(prompt) :]
        else:
            prompt = ""
            response = apply_chat_template(data[response_key], tokenize=False)
    else:
        if prompt_key:
            prompt = data[prompt_key]
            if input_template:
                prompt = input_template.format(prompt)
        else:
            prompt = ""
        response = data[response_key]

    # margin loss
    reward = data["reward"] 

    return prompt, response, reward

"""
Process multi-class reward dataset
"""
def preprocess_multiclass_data(
    data,
    input_template=None,
    prompt_key="prompt",
    response_key="response",
    apply_chat_template=None,
    is_dpo=False,
) -> str:
    if apply_chat_template:
        if prompt_key:
            prompt = apply_chat_template(data[prompt_key], tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(data[prompt_key] + data[response_key], tokenize=False)[len(prompt) :]
        else:
            prompt = ""
            response = apply_chat_template(data[response_key], tokenize=False)
    else:
        if prompt_key:
            prompt = data[prompt_key]
            if input_template:
                prompt = input_template.format(prompt)
        else:
            prompt = ""
        response = data[response_key]

    # margin loss
    repetition_reward = data["repetition_penalty"] 
    positive_reward = data["positive_reward"]
    negative_reward = data["negative_reward"]
    return prompt, response, repetition_reward, positive_reward, negative_reward

"""
Pairwise x, y1, y2 input
"""
def preprocess_data(
    data,
    input_template=None,
    prompt_key=None,
    chosen_key="chosen",
    rejected_key="rejected",
    apply_chat_template=None,
    is_dpo=False,
) -> str:
    if apply_chat_template:
        if prompt_key:
            # Add a custom sytem prompt
            custom_system_message = data.get("system_message", data.get("system_prompt", "You are a helpful assistant."))

            custom_system_message += " Keep your response to 2-3 sentences."
            # Using a custom system prompt is my added code
            system_message = [{"role": "system", "content": custom_system_message}]
            prompt = apply_chat_template(system_message + [{"role": "user", "content": data[prompt_key]}], tokenize=False, add_generation_prompt=True)
            chosen = apply_chat_template([{"role": "system", "content": custom_system_message}] + [{"role": "user", "content": data[prompt_key]}] + [{"role": "assistant", "content": data[chosen_key]}], tokenize=False)[len(prompt) :]
            rejected = apply_chat_template([{"role": "system", "content": custom_system_message}] + [{"role": "user", "content": data[prompt_key]}] + [{"role": "assistant", "content": data[rejected_key]}], tokenize=False)[len(prompt) :]
        else:
            prompt = ""
            chosen = apply_chat_template(data[chosen_key], tokenize=False)
            rejected = apply_chat_template(data[rejected_key], tokenize=False)

            if is_dpo:
                prompt = apply_chat_template(data[chosen_key][:-1], tokenize=False, add_generation_prompt=True)
                chosen = chosen[len(prompt) :]
                rejected = rejected[len(prompt) :]
    else:
        if prompt_key:
            prompt = data[prompt_key]
            if input_template:
                prompt = input_template.format(prompt)
        else:
            prompt = ""
        chosen = data[chosen_key]
        rejected = data[rejected_key]

    # margin loss
    margin = data["margin"] if exist_and_not_none(data, "margin") else 0

    return prompt, chosen, rejected, margin

class MultiClassRewardEvalDataset(Dataset):
    """
    Dataset for multi reward model (3 rewards per prompt)

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        is_dpo=False,
        num_processors=8,
        multiple_of=1,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of

        # chat_template
        self.input_template = input_template
        self.prompt_key = getattr(self.strategy.args, "prompt_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template
        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )

        # Filter out None values if necessary
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.repetition_rewards = processed_dataset["repetition_reward"]
        self.positive_rewards = processed_dataset["positive_reward"]
        self.negative_rewards = processed_dataset["negative_reward"]

    def process_data(self, data):
        prompt, response, repetition_reward, positive_reward, negative_reward = preprocess_multiclass_data(
            data,
            self.input_template,
            prompt_key="prompt",
            response_key="response",
            apply_chat_template=self.apply_chat_template,
        )

        return {
            "prompt": prompt,
            "response": response,
            "repetition_reward": repetition_reward,
            "positive_reward": positive_reward,
            "negative_reward": negative_reward
        }

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt, response, repetition_reward, positive_reward, negative_reward = self.prompts[idx], self.responses[idx], self.repetition_rewards[idx], self.positive_rewards[idx], self.negative_rewards[idx]

        response = (prompt + response).rstrip("\n")
        if not response.endswith(self.tokenizer.eos_token):
            response += " " + self.tokenizer.eos_token
        response_token = self.tokenizer(
            response,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        # to avoid EOS_token truncation
        response_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        response_token["attention_mask"][0][-1] = True
    
        return (
            response_token["input_ids"],
            response_token["attention_mask"],
            repetition_reward,
            positive_reward,
            negative_reward,
            {"prompt": prompt, "response": response}
        )

    def collate_fn(self, item_list):
        response_ids = []
        response_masks = []
        repetition_rewards = []
        positive_rewards = []
        negative_rewards = []
        info_list = []
        for response_id, response_mask, repetition_reward, positive_reward, negative_reward, info in item_list:
            response_ids.append(response_id)
            response_masks.append(response_mask)
            repetition_rewards.append(repetition_reward)
            positive_rewards.append(positive_reward)
            negative_rewards.append(negative_reward)
            info_list.append(info)

        padding_side = "left"
        response_ids = zero_pad_sequences(response_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        response_masks = zero_pad_sequences(response_masks, side=padding_side)
        return response_ids, response_masks, repetition_rewards, positive_rewards, negative_rewards, info_list

    def packing_collate_fn(self, item_list):
        repetition_rewards = []
        positive_rewards = []
        negative_rewards = []

        chosen_ids = []
        chosen_att_masks = []
        chosen_seq_lens = []
        info_list = []
        index = 1
        for chosen_id, chosen_mask, repetition_reward, positive_reward, negative_reward, info in item_list:
            chosen_ids.append(chosen_id.flatten())
            chosen_att_masks.append(torch.full_like(chosen_id.flatten(), index))
            chosen_seq_lens.append(len(chosen_id.flatten()))
            repetition_rewards.append(repetition_reward)
            positive_rewards.append(positive_reward)
            negative_rewards.append(negative_reward)
            info_list.append(info)
            index += 1

        packed_input_ids = torch.cat(chosen_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(chosen_att_masks, dim=0).unsqueeze(0)
        packed_seq_lens = chosen_seq_lens 

        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0:
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return packed_input_ids, packed_attention_masks, packed_seq_lens, repetition_rewards, positive_rewards, negative_rewards, info_list


class MultiClassRewardDataset(Dataset):
    """
    Dataset for multi reward model (3 rewards per prompt)

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        is_dpo=False,
        num_processors=8,
        multiple_of=1,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of

        # chat_template
        self.input_template = input_template
        self.prompt_key = getattr(self.strategy.args, "prompt_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template
        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )

        # Filter out None values if necessary
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.repetition_rewards = processed_dataset["repetition_reward"]
        self.positive_rewards = processed_dataset["positive_reward"]
        self.negative_rewards = processed_dataset["negative_reward"]

    def process_data(self, data):
        prompt, response, repetition_reward, positive_reward, negative_reward = preprocess_multiclass_data(
            data,
            self.input_template,
            prompt_key="prompt",
            response_key="response",
            apply_chat_template=self.apply_chat_template,
        )

        return {
            "prompt": prompt,
            "response": response,
            "repetition_reward": repetition_reward,
            "positive_reward": positive_reward,
            "negative_reward": negative_reward
        }

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt, response, repetition_reward, positive_reward, negative_reward = self.prompts[idx], self.responses[idx], self.repetition_rewards[idx], self.positive_rewards[idx], self.negative_rewards[idx]

        response = (prompt + response).rstrip("\n")
        if not response.endswith(self.tokenizer.eos_token):
            response += " " + self.tokenizer.eos_token
        response_token = self.tokenizer(
            response,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        # to avoid EOS_token truncation
        response_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        response_token["attention_mask"][0][-1] = True
    
        return (
            response_token["input_ids"],
            response_token["attention_mask"],
            repetition_reward,
            positive_reward,
            negative_reward
        )

    def collate_fn(self, item_list):
        response_ids = []
        response_masks = []
        repetition_rewards = []
        positive_rewards = []
        negative_rewards = []
        for response_id, response_mask, repetition_reward, positive_reward, negative_reward in item_list:
            response_ids.append(response_id)
            response_masks.append(response_mask)
            repetition_rewards.append(repetition_reward)
            positive_rewards.append(positive_reward)
            negative_rewards.append(negative_reward)

        padding_side = "left"
        response_ids = zero_pad_sequences(response_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        response_masks = zero_pad_sequences(response_masks, side=padding_side)
        return response_ids, response_masks, repetition_rewards, positive_rewards, negative_rewards

    def packing_collate_fn(self, item_list):
        repetition_rewards = []
        positive_rewards = []
        negative_rewards = []

        chosen_ids = []
        chosen_att_masks = []
        chosen_seq_lens = []
        
        index = 1
        for chosen_id, chosen_mask, repetition_reward, positive_reward, negative_reward in item_list:
            chosen_ids.append(chosen_id.flatten())
            chosen_att_masks.append(torch.full_like(chosen_id.flatten(), index))
            chosen_seq_lens.append(len(chosen_id.flatten()))
            repetition_rewards.append(repetition_reward)
            positive_rewards.append(positive_reward)
            negative_rewards.append(negative_reward)
            index += 1

        packed_input_ids = torch.cat(chosen_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(chosen_att_masks, dim=0).unsqueeze(0)
        packed_seq_lens = chosen_seq_lens 

        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0:
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return packed_input_ids, packed_attention_masks, packed_seq_lens, repetition_rewards, positive_rewards, negative_rewards

class SingularRewardEvalDataset(Dataset):
    """
    Dataset for non-pairwise reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        is_dpo=False,
        num_processors=8,
        multiple_of=1,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of

        # chat_template
        self.input_template = input_template
        self.prompt_key = getattr(self.strategy.args, "prompt_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template
            print("!! Inside apply chat template...", self.apply_chat_template)
        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )

        # Filter out None values if necessary
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.rewards = processed_dataset["reward"]

    def process_data(self, data):
        prompt, response, reward = preprocess_nonpair_data(
            data,
            self.input_template,
            prompt_key="prompt",
            response_key="response",
            apply_chat_template=self.apply_chat_template,
        )

        return {
            "prompt": prompt,
            "response": response,
            "reward": reward,
        }

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt, response, reward = self.prompts[idx], self.responses[idx], self.rewards[idx]

        response = (prompt + response).rstrip("\n")
        if not response.endswith(self.tokenizer.eos_token):
            response += " " + self.tokenizer.eos_token
        response_token = self.tokenizer(
            response,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        # to avoid EOS_token truncation
        response_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        response_token["attention_mask"][0][-1] = True
    
        return (
            response_token["input_ids"],
            response_token["attention_mask"],
            reward,
            {"prompt": prompt, "response": response, "reward": reward}
        )

    def collate_fn(self, item_list):
        response_ids = []
        response_masks = []
        rewards = []
        info_lists = []
        for response_id, response_mask, reward, info in item_list:
            response_ids.append(response_id)
            response_masks.append(response_mask)
            rewards.append(reward)
            info_lists.append(info)

        padding_side = "left"
        response_ids = zero_pad_sequences(response_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        response_masks = zero_pad_sequences(response_masks, side=padding_side)
        return response_ids, response_masks, rewards, info_lists

    def packing_collate_fn(self, item_list):
        rewards = []
        info_lists = []
        chosen_ids = []
        chosen_att_masks = []
        chosen_seq_lens = []
        
        index = 1
        for chosen_id, chosen_mask, reward, info in item_list:
            chosen_ids.append(chosen_id.flatten())
            chosen_att_masks.append(torch.full_like(chosen_id.flatten(), index))
            chosen_seq_lens.append(len(chosen_id.flatten()))
            rewards.append(reward)
            info_lists.apend(info)
            index += 1

        packed_input_ids = torch.cat(chosen_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(chosen_att_masks, dim=0).unsqueeze(0)
        packed_seq_lens = chosen_seq_lens 

        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0:
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return packed_input_ids, packed_attention_masks, packed_seq_lens, rewards, info_lists



class SingularRewardDataset(Dataset):
    """
    Dataset for non-pairwise reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        is_dpo=False,
        num_processors=8,
        multiple_of=1,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of

        # chat_template
        self.input_template = input_template
        self.prompt_key = getattr(self.strategy.args, "prompt_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template
            print("!! Inside apply chat template...", self.apply_chat_template)
        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )

        # Filter out None values if necessary
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.rewards = processed_dataset["reward"]

    def process_data(self, data):
        prompt, response, reward = preprocess_nonpair_data(
            data,
            self.input_template,
            prompt_key="prompt",
            response_key="response",
            apply_chat_template=self.apply_chat_template,
        )

        return {
            "prompt": prompt,
            "response": response,
            "reward": reward,
        }

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt, response, reward = self.prompts[idx], self.responses[idx], self.rewards[idx]

        response = (prompt + response).rstrip("\n")
        if not response.endswith(self.tokenizer.eos_token):
            response += " " + self.tokenizer.eos_token
        response_token = self.tokenizer(
            response,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        # to avoid EOS_token truncation
        response_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        response_token["attention_mask"][0][-1] = True
    
        return (
            response_token["input_ids"],
            response_token["attention_mask"],
            reward
        )

    def collate_fn(self, item_list):
        response_ids = []
        response_masks = []
        rewards = []
        for response_id, response_mask, reward in item_list:
            response_ids.append(response_id)
            response_masks.append(response_mask)
            rewards.append(reward)


        padding_side = "left"
        response_ids = zero_pad_sequences(response_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        response_masks = zero_pad_sequences(response_masks, side=padding_side)
        return response_ids, response_masks, rewards

    def packing_collate_fn(self, item_list):
        rewards = []

        chosen_ids = []
        chosen_att_masks = []
        chosen_seq_lens = []
        
        index = 1
        for chosen_id, chosen_mask, reward in item_list:
            chosen_ids.append(chosen_id.flatten())
            chosen_att_masks.append(torch.full_like(chosen_id.flatten(), index))
            chosen_seq_lens.append(len(chosen_id.flatten()))
            rewards.append(reward)
            index += 1

        packed_input_ids = torch.cat(chosen_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(chosen_att_masks, dim=0).unsqueeze(0)
        packed_seq_lens = chosen_seq_lens 

        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0:
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return packed_input_ids, packed_attention_masks, packed_seq_lens, rewards

class RewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        is_dpo=False,
        num_processors=8,
        multiple_of=1,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of

        # chat_template
        self.input_template = input_template
        self.prompt_key = getattr(self.strategy.args, "prompt_key", None)
        self.chosen_key = getattr(self.strategy.args, "chosen_key", None)
        self.rejected_key = getattr(self.strategy.args, "rejected_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )

        # Filter out None values if necessary
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.chosens = processed_dataset["chosen"]
        self.rejects = processed_dataset["reject"]
        self.extras = processed_dataset["extra"]

    def process_data(self, data):
        prompt, chosen, reject, margin = preprocess_data(
            data,
            self.input_template,
            self.prompt_key,
            self.chosen_key,
            self.rejected_key,
            self.apply_chat_template,
            self.is_dpo,
        )

        if self.is_dpo:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            # Filter the sample whose length is greater than max_length (2 for answer length)
            if prompt_ids_len >= self.max_length - 2:
                prompt = None

        return {
            "prompt": prompt,
            "chosen": chosen,
            "reject": reject,
            "extra": prompt_ids_len if self.is_dpo else margin,
        }

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, chosen, reject, extra = self.prompts[idx], self.chosens[idx], self.rejects[idx], self.extras[idx]

        chosen = (prompt + chosen).rstrip("\n")
        if not chosen.endswith(self.tokenizer.eos_token):
            chosen += " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        reject = (prompt + reject).rstrip("\n")
        if not reject.endswith(self.tokenizer.eos_token):
            reject += " " + self.tokenizer.eos_token
        reject_token = self.tokenizer(
            reject,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            reject_token["input_ids"],
            reject_token["attention_mask"],
            extra,
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        extras = []
        for chosen_id, chosen_mask, reject_id, rejects_mask, extra in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            extras.append(extra)

        if self.is_dpo:
            padding_side = "right"
        else:
            padding_side = "left"
        chosen_ids = zero_pad_sequences(chosen_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks, side=padding_side)
        reject_ids = zero_pad_sequences(reject_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks, side=padding_side)
        return chosen_ids, chosen_masks, reject_ids, rejects_masks, extras

    def packing_collate_fn(self, item_list):
        extras = []

        chosen_ids = []
        chosen_att_masks = []
        chosen_seq_lens = []
        rejected_ids = []
        rejected_att_masks = []
        rejected_seq_lens = []
        index = 1
        for chosen_id, chosen_mask, reject_id, rejects_mask, extra in item_list:
            chosen_ids.append(chosen_id.flatten())
            chosen_att_masks.append(torch.full_like(chosen_id.flatten(), index))
            chosen_seq_lens.append(len(chosen_id.flatten()))
            extras.append(extra)

            rejected_ids.append(reject_id.flatten())
            rejected_att_masks.append(torch.full_like(reject_id.flatten(), index + len(item_list)))
            rejected_seq_lens.append(len(reject_id.flatten()))
            index += 1

        packed_input_ids = torch.cat(chosen_ids + rejected_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(chosen_att_masks + rejected_att_masks, dim=0).unsqueeze(0)
        packed_seq_lens = chosen_seq_lens + rejected_seq_lens

        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0:
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return packed_input_ids, packed_attention_masks, packed_seq_lens, extras
