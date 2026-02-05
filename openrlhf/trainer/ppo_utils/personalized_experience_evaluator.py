import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import math 
import numpy as np
import ray
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm

import re

from openrlhf.models.actor import Actor
from openrlhf.models.ring_attn_utils import pad_sequences, unpad_sequences
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray
from openrlhf.models import LogExpLoss, PairWiseLoss

from transformers.trainer import get_scheduler

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.base_action_log_probs = to(self.base_action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.base_action_log_probs = pin_memory(self.base_action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.kl = pin_memory(self.kl)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        return self


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    prompts: the prompts used to generate responses
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]  
    action_mask: Optional[torch.BoolTensor] 
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor] 
    response_length: torch.Tensor
    total_length: torch.Tensor
    prompts: list[str]
    labels: Optional[list[str]] = None
    rejected: Optional[List[str]] = None
    chosen: Optional[List[str]] = None
    reward_prompts: Optional[List[str]]= None
    train_prompts: Optional[List[str]]= None
    eval_prompts: Optional[List[str]]= None
    train_rejected: Optional[List[str]]= None
    train_chosen: Optional[List[str]] = None 
    eval_rejected: Optional[List[str]] = None
    eval_chosen: Optional[List[str]] = None
    pad_len: Optional[int] = None


class PersonalizedExperienceEvaluator(ABC):
    """
    Personalized experience maker used for PPO summarizer.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        reward_tokenizer,
        reward_optim,
        reward_scheduler,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: Union[list[str], str] = None,
        reward_fn=None,
        train_reward=False,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.reward_tokenizer = reward_tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.summary_logs = []
        self.prediction_acc_logs = []
        self.prediction_rew_logs = []
        self.eval_accs = []
        self.eval_rews = []
        self.fin_eval_accs = []
        self.fin_eval_rews = []
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator
        self.prediction_reward_func = PairWiseLoss()
        self._train_reward = train_reward
        self.prediction_acc = 0.
        self.reward_optim = reward_optim  
        num_update_steps_per_epoch = self.strategy.args.rollout_batch_size 
        max_steps = math.ceil(self.strategy.args.max_epochs * num_update_steps_per_epoch)

        self.reward_scheduler = reward_scheduler

        # custom reward func for reinforced finetuning
        self.custom_reward_func = None
        remote_rm_url = [remote_rm_url] if isinstance(remote_rm_url, str) else remote_rm_url
        if remote_rm_url and remote_rm_url[0].endswith(".py"):
            print(f"Loading custom `reward_func(queries, prompts, labels)` from {remote_rm_url[0]}")
            import importlib.util

            spec = importlib.util.spec_from_file_location("reward_func", remote_rm_url[0])
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.custom_reward_func = reward_module.reward_func

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    def train_reward_model(self, concatenated_summaries, concatenated_train_prompts, concatenated_train_rejected, concatenated_train_chosen):
        self.reward_model.train()
        torch.cuda.empty_cache()
        device = torch.cuda.current_device()
        reward_loss_fn = PairWiseLoss()

        # split into batches before training:
        total_samples = len(concatenated_summaries)
        batch_size = 2
        for start_idx in range(0, total_samples, batch_size):
            end_idx = start_idx + batch_size        
            chosen_chats = ["Based on the following summary about the user from their past conversation, respond to the user's request in a way that aligns with the user's preference. Summary: " + summary + "\nStart of a new conversation: User: " + prompt + "\nAssistant: " + response + " " + self.reward_tokenizer.eos_token for summary, response, prompt in zip(concatenated_summaries[start_idx: end_idx], concatenated_train_chosen[start_idx: end_idx], concatenated_train_prompts[start_idx: end_idx])]
            rejected_chats = ["Based on the following summary about the user from their past conversation, respond to the user's request in a way that aligns with the user's preference. Summary: " + summary + "\nStart of a new conversation: User: " + prompt + "\nAssistant: " + response + " " + self.reward_tokenizer.eos_token for summary, response, prompt in zip(concatenated_summaries[start_idx: end_idx], concatenated_train_rejected[start_idx: end_idx], concatenated_train_prompts[start_idx: end_idx])]
            
            chosen_chats = [chat.rstrip("\n") for chat in chosen_chats]
            rejected_chats = [chat.rstrip("\n") for chat in rejected_chats]
            # Tokenization with padding and truncation
            chosen_tokens = self.reward_tokenizer(
                chosen_chats,
                max_length=1024,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            rejected_tokens = self.reward_tokenizer(
                rejected_chats,
                max_length=1024,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            chosen_input_ids = chosen_tokens["input_ids"].to(device)
            chosen_attention_mask = chosen_tokens["attention_mask"].to(device)

            rejected_input_ids = rejected_tokens["input_ids"].to(device)
            rejected_attention_mask = rejected_tokens["attention_mask"].to(device)

            # Add EOS token to the sequences
            eos_token_id = self.reward_tokenizer.eos_token_id

            # Create tensors for EOS token
            eos_tensor = torch.full((chosen_tokens["input_ids"].shape[0], 1), eos_token_id, dtype=torch.long).to(device)
            chosen_tokens["input_ids"] = torch.cat([chosen_input_ids, eos_tensor], dim=1)

            eos_mask = torch.ones((chosen_tokens["attention_mask"].shape[0], 1), dtype=torch.long).to(device)
            chosen_tokens["attention_mask"] = torch.cat([chosen_attention_mask, eos_mask], dim=1)

            # Do the same for rejected tokens
            eos_tensor = torch.full((rejected_tokens["input_ids"].shape[0], 1), eos_token_id, dtype=torch.long).to(device)
            rejected_tokens["input_ids"] = torch.cat([rejected_input_ids, eos_tensor], dim=1)

            eos_mask = torch.ones((rejected_tokens["attention_mask"].shape[0], 1), dtype=torch.long).to(device)
            rejected_tokens["attention_mask"] = torch.cat([rejected_attention_mask, eos_mask], dim=1)

            # Assign directly (no need for list comprehension)
            chosen_input_ids = chosen_tokens["input_ids"]
            rejected_input_ids = rejected_tokens["input_ids"]
            chosen_attention_mask = chosen_tokens["attention_mask"]
            rejected_attention_mask = rejected_tokens["attention_mask"]

            # Combine chosen and rejected inputs into a single batch
            input_ids_batch = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
            attention_mask_batch = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
            # Forward pass
            all_values, output = self.reward_model(input_ids=input_ids_batch, attention_mask=attention_mask_batch, return_output=True)
            # Separate chosen and rejected outputs
            chosen_r = all_values[: len(chosen_input_ids)]
            rejected_r = all_values[len(chosen_input_ids):]
            preference_loss = self.prediction_reward_func(chosen_r, rejected_r, None)
            self.strategy.backward(preference_loss, self.reward_model, self.reward_optim)
            self.strategy.optimizer_step(self.reward_optim, self.reward_model, self.reward_scheduler)
        torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluate_reward_model(self, summaries, prompts, rejected, chosen, return_all=False):
        self.reward_model.eval()
        chosen_chats = ["Based on the following summary about the user from their past conversation, respond to the user's request in a way that aligns with the user's preference. Summary: " + summary + "\nStart of a new conversation: User: " + prompt + "\nAssistant: " + response + " " + self.reward_tokenizer.eos_token for summary, response, prompt in zip(summaries, chosen, prompts)]
        rejected_chats = ["Based on the following summary about the user from their past conversation, respond to the user's request in a way that aligns with the user's preference. Summary: " + summary + "\nStart of a new conversation: User: " + prompt + "\nAssistant: " + response + " " + self.reward_tokenizer.eos_token for summary, response, prompt in zip(summaries, rejected, prompts)]
        
        chosen_chats = [chat.rstrip("\n") for chat in chosen_chats]
        rejected_chats = [chat.rstrip("\n") for chat in rejected_chats]

        # Tokenization with padding and truncation
        chosen_tokens = self.reward_tokenizer(
            chosen_chats,
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.reward_model.device)

        rejected_tokens = self.reward_tokenizer(
            rejected_chats,
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.reward_model.device)

        # Add EOS token to the sequences
        eos_token_id = self.reward_tokenizer.eos_token_id

        # Create tensors for EOS token
        eos_tensor = torch.full((chosen_tokens["input_ids"].shape[0], 1), eos_token_id, dtype=torch.long).to(self.reward_model.device)
        chosen_tokens["input_ids"] = torch.cat([chosen_tokens["input_ids"], eos_tensor], dim=1)

        eos_mask = torch.ones((chosen_tokens["attention_mask"].shape[0], 1), dtype=torch.long).to(self.reward_model.device)
        chosen_tokens["attention_mask"] = torch.cat([chosen_tokens["attention_mask"], eos_mask], dim=1)

        # Do the same for rejected tokens
        eos_tensor = torch.full((rejected_tokens["input_ids"].shape[0], 1), eos_token_id, dtype=torch.long).to(self.reward_model.device)
        rejected_tokens["input_ids"] = torch.cat([rejected_tokens["input_ids"], eos_tensor], dim=1)

        eos_mask = torch.ones((rejected_tokens["attention_mask"].shape[0], 1), dtype=torch.long).to(self.reward_model.device)
        rejected_tokens["attention_mask"] = torch.cat([rejected_tokens["attention_mask"], eos_mask], dim=1)

        # Assign directly (no need for list comprehension)
        chosen_input_ids = chosen_tokens["input_ids"]
        rejected_input_ids = rejected_tokens["input_ids"]
        chosen_attention_mask = chosen_tokens["attention_mask"]
        rejected_attention_mask = rejected_tokens["attention_mask"]

        # Combine chosen and rejected inputs into a single batch
        input_ids_batch = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        attention_mask_batch = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)

        # Forward pass
        outputs = self.reward_model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
        # Separate chosen and rejected outputs
        chosen_r = outputs[: len(chosen_input_ids)]
        rejected_r = outputs[len(chosen_input_ids):]
        return rejected_r, chosen_r

    @torch.no_grad()
    def compute_reward_model(self, experiences):
        for experience in experiences:
            summaries = experience.info["summary"]
            chosen = experience.info["chosen"]
            rejected = experience.info["rejected"]
            reward_prompts = experience.info["reward_prompt"]
            chosen_chats = ["Based on the following summary about the user from their past conversation, respond to the user's request in a way that aligns with the user's preference. Summary: " + summary + "\nStart of a new conversation: User: " + prompt + "\nAssistant: " + response + " " + self.reward_tokenizer.eos_token for summary, response, prompt in zip(summaries, chosen, reward_prompts)]
            rejected_chats = ["Based on the following summary about the user from their past conversation, respond to the user's request in a way that aligns with the user's preference. Summary: " + summary + "\nStart of a new conversation: User: " + prompt + "\nAssistant: " + response + " " + self.reward_tokenizer.eos_token for summary, response, prompt in zip(summaries, rejected, reward_prompts)]
            
            chosen_chats = [chat.rstrip("\n") for chat in chosen_chats]
            rejected_chats = [chat.rstrip("\n") for chat in rejected_chats]

            # Tokenization with padding and truncation
            chosen_tokens = self.reward_tokenizer(
                chosen_chats,
                max_length=16384,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.reward_model.device)

            rejected_tokens = self.reward_tokenizer(
                rejected_chats,
                max_length=16384,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.reward_model.device)

            # Add EOS token to the sequences
            eos_token_id = self.reward_tokenizer.eos_token_id

            # Create tensors for EOS token
            eos_tensor = torch.full((chosen_tokens["input_ids"].shape[0], 1), eos_token_id, dtype=torch.long).to(self.reward_model.device)
            chosen_tokens["input_ids"] = torch.cat([chosen_tokens["input_ids"], eos_tensor], dim=1)

            eos_mask = torch.ones((chosen_tokens["attention_mask"].shape[0], 1), dtype=torch.long).to(self.reward_model.device)
            chosen_tokens["attention_mask"] = torch.cat([chosen_tokens["attention_mask"], eos_mask], dim=1)

            # Do the same for rejected tokens
            eos_tensor = torch.full((rejected_tokens["input_ids"].shape[0], 1), eos_token_id, dtype=torch.long).to(self.reward_model.device)
            rejected_tokens["input_ids"] = torch.cat([rejected_tokens["input_ids"], eos_tensor], dim=1)

            eos_mask = torch.ones((rejected_tokens["attention_mask"].shape[0], 1), dtype=torch.long).to(self.reward_model.device)
            rejected_tokens["attention_mask"] = torch.cat([rejected_tokens["attention_mask"], eos_mask], dim=1)

            # Assign directly (no need for list comprehension)
            chosen_input_ids = chosen_tokens["input_ids"]
            rejected_input_ids = rejected_tokens["input_ids"]
            chosen_attention_mask = chosen_tokens["attention_mask"]
            rejected_attention_mask = rejected_tokens["attention_mask"]

            # Combine chosen and rejected inputs into a single batch
            input_ids_batch = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
            attention_mask_batch = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)

            # Forward pass
            outputs = self.reward_model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
            # Separate chosen and rejected outputs
            chosen_r = outputs[: len(chosen_input_ids)]
            rejected_r = outputs[len(chosen_input_ids):]
            # change to accuracy
            acc = (chosen_r > rejected_r).float().mean().item()
            r = torch.tensor([- self.prediction_reward_func(c, r) for c, r in zip(chosen_r, rejected_r)], dtype=torch.float).to(self.reward_model.device)
            self.prediction_acc_logs.append(acc)
            self.prediction_rew_logs.append(r.mean().item())
            experience.info["reward"] = r
        return experiences 

    @torch.no_grad()
    def make_experience_list(
        self, all_prompts: Union[str, List[str]], all_rejected, all_chosen, reward_prompts, train_prompts, eval_prompts, train_rejected, train_chosen, eval_rejected, eval_chosen, **generate_kwargs
    ) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        args = self.strategy.args
        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            torch.distributed.barrier()
            torch.cuda.synchronize()
        # generate responses
        self.reward_model.eval()
        if self.strategy.ring_attn_group is not None:
            # Only rank 0 in the ring attention group executes the generation function, and then broadcasts it to all other ranks.
            if self.strategy.ring_attn_rank == 0:
                samples_list = self.generate_samples(all_prompts, all_rejected, all_chosen, reward_prompts, train_prompts, eval_prompts, train_rejected, train_chosen, eval_rejected, eval_chosen, **generate_kwargs)
                dist.broadcast_object_list(samples_list, src=dist.get_rank(), group=self.strategy.ring_attn_group)
            else:
                world_size = torch.distributed.get_world_size() // args.ring_attn_size
                samples_list = [None] * (
                    args.rollout_batch_size * args.n_samples_per_prompt // world_size // args.micro_rollout_batch_size
                )
                dist.broadcast_object_list(
                    samples_list, src=self.strategy.ring_attn_ranks[0], group=self.strategy.ring_attn_group
                )
        else:
            samples_list = self.generate_samples(all_prompts, all_rejected, all_chosen, reward_prompts, train_prompts, eval_prompts, train_rejected, train_chosen, eval_rejected, eval_chosen, **generate_kwargs)

        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")
        torch.distributed.barrier()
        torch.cuda.synchronize()

        experiences = []
        for samples in tqdm(
            samples_list,
            desc="make_experience",
            disable=not self.strategy.is_rank_0(),
        ):
            experiences.append(self.make_experience(samples).to_device("cpu"))

        self.reward_model.eval()
        # technically need to evaluate the reward model after training ... (so need to split the reward computation part of the code from make_experience)
        concatenated_rejected = [experience.info["rejected"] for experience in experiences]
        concatenated_chosen = [experience.info["chosen"] for experience in experiences]
        experiences = self.compute_reward_model(experiences)
        experiences, rewards = self.process_experiences(experiences)

        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")
            num_actions = experience.info["num_actions"]
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm"]:
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            del experience.info["chosen"]
            del experience.info["rejected"]
            experience.to_device("cpu")
        
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_rejected: List[str], all_chosen: List[str], reward_prompts: List[str], train_prompts: List[str], eval_prompts: List[str], train_rejected: List[str], train_chosen: List[str], eval_rejected: List[str], eval_chosen: List[str], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_rejected = sum([[rejected] * args.n_samples_per_prompt for rejected in all_rejected], [])
        all_chosen = sum([[chosen] * args.n_samples_per_prompt for chosen in all_chosen], [])
        all_reward_prompts = sum([[reward_prompt] * args.n_samples_per_prompt for reward_prompt in reward_prompts], [])
        all_train_prompts = sum([[train_prompt] * args.n_samples_per_prompt for train_prompt in train_prompts], [])
        all_eval_prompts = sum([[eval_prompt] * args.n_samples_per_prompt for eval_prompt in eval_prompts], [])
        all_train_rejected = sum([[rejected] * args.n_samples_per_prompt for rejected in train_rejected], [])
        all_train_chosen = sum([[chosen] * args.n_samples_per_prompt for chosen in train_chosen], [])
        all_eval_rejected = sum([[rejected] * args.n_samples_per_prompt for rejected in eval_rejected], [])
        all_eval_chosen = sum([[chosen] * args.n_samples_per_prompt for chosen in eval_chosen], [])
        samples_list = []
        # halting here? 
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            print(i, len(all_prompts))
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            rejected = all_rejected[i : i + args.micro_rollout_batch_size]
            chosen = all_chosen[i : i + args.micro_rollout_batch_size]
            reward_prompts = all_reward_prompts[i : i + args.micro_rollout_batch_size]
            train_prompts = all_train_prompts[i : i + args.micro_rollout_batch_size]
            eval_prompts = all_eval_prompts[i : i + args.micro_rollout_batch_size]
            train_rejected = all_train_rejected[i : i + args.micro_rollout_batch_size]
            train_chosen = all_train_chosen[i : i + args.micro_rollout_batch_size]
            eval_rejected = all_eval_rejected[i : i + args.micro_rollout_batch_size]
            eval_chosen = all_eval_chosen[i : i + args.micro_rollout_batch_size]
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                prompts=prompts,
                rejected=rejected,
                chosen=chosen,
                reward_prompts=reward_prompts,
                eval_prompts=eval_prompts,
                train_prompts=train_prompts,
                train_rejected=train_rejected,
                train_chosen=train_chosen,
                eval_rejected=eval_rejected,
                eval_chosen=eval_chosen
            )
            samples_list.append(samples)
        return samples_list

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        if self.initial_model is not None:
            self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        if self.critic is not None:
            self.critic.eval()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        if self.initial_model is not None:
            base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)
        else:
            base_action_log_probs = None

        # values
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask)
        else:
            value = None

        # Need to turn sequences.cpu() into queries
        summaries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=True)
        # sometimes th generated summaries need to be processed differently
        summaries = [summary.split("\nassistant\n")[-1] for summary in summaries]
        self.reward_model.eval()
        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            if self.custom_reward_func:
                r = self.custom_reward_func(queries, samples.prompts, samples.labels).to(
                    device=action_log_probs.device
                )
            else:
                r = remote_rm_fn(
                    self.remote_rm_url, queries=queries, prompts=samples.prompts, labels=samples.labels
                ).to(device=action_log_probs.device)
        else:
            r = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)

        if (self.initial_model is not None) and (not self.strategy.args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                kl_estimator=self.strategy.args.kl_estimator,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)

        # TODO: need to append train_prompts, eval_prompts
        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
            "summary": summaries, 
            "chosen": samples.chosen,
            "rejected": samples.rejected,
            "train_rejected": samples.train_rejected,
            "train_chosen": samples.train_chosen,
            "eval_rejected": samples.eval_rejected,
            "eval_chosen": samples.eval_chosen,
            "reward_prompt": samples.reward_prompts,
            "train_prompt": samples.train_prompts,
            "eval_prompt": samples.eval_prompts,
        }
        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            base_action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience]) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args
        # reward shaping for rloo and reinforce_baseline
        if args.advantage_estimator == "rloo":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "reinforce_baseline":
            # REINFORCE++-baseline removed the / std and K3 kl loss in GRPO.
            # `/ std` is not needed in RL variance reduction theory, and `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = rewards - rewards.mean(-1, keepdim=True)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "group_norm":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        # default rewards
        return experiences, [experience.info["reward"] for experience in experiences]

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns
