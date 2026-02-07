# Learning to summarize user information from personalized reinforcement learning from human feedback 
## Preference Learning Using Summarization (PLUS): 
[arxiv link:](https://arxiv.org/abs/2507.13579)

<img width="1212" height="409" alt="plus_revised_figure" src="https://github.com/user-attachments/assets/6bf06038-435f-4c95-a4e0-a7fd0ea5c57a" />

- Our code is built on top of [OpenRLHF (Hu et al., 2024)](https://github.com/OpenRLHF/OpenRLHF). Please refer to the OpenRLHF repo for version updates.
- The dataset requires the following fields:
```
context, context_messages,
rejected, chosen, reward_prompt, # for summarizer training 
train_prompt, train_rejected, train_chosen, # for reward model training 
eval_prompt, eval_rejected, eval_chosen, # for reward model evaluation a held-out example (not necessary)
```
- For training stability, we recommend using the same train_prompt and reward_prompt; but to avoid overfitting, consdier using different prompts for the reward model and the summarizer.
- Context/context_messages end with `"Start your response with ##Summary"`, so the summarizer output can be easily delimited and used to condition the reward model with.

- To run PLUS with PPO:

```
deepspeed 
--module openrlhf.cli.train_ppo
--pretrain Qwen/Qwen2.5-3B-Instruct
--reward_pretrain Qwen/Qwen2.5-1.5B-Instruct
--save_path ./checkpoint/
--save_steps 10
--logging_steps 1
--eval_steps -1
--init_kl_coef 0.01
--micro_train_batch_size 1
--train_batch_size 128
--micro_rollout_batch_size 2
--rollout_batch_size 256
--max_epochs 1
--prompt_max_len 15360
--generate_max_len 256
--zero_stage 2
--bf16
--actor_learning_rate 5e-7
--critic_learning_rate 9e-6
--prompt_data json@
--input_key context_messages
--apply_chat_template
--max_samples 100000
--normalize_reward
--flash_attn
--gradient_checkpointing
--save_hf_ckpt
--num_episodes 1
--debug_log_path tmp.pkl
--model_save_path tmp_model
--train_reward
```

- This is an example of what the reward model might see during training (i.e., $z$ = summarizer output is concatenated with the new user's query):
```
[{'role': 'user', 'content': 'Here is the summary of my preferences based on previous conversations: The user prefers clear and complete answers reflecting all provided details. Preferred phrasing includes "Lord Esketh and his wife, Edwina," connecting the full description of the attendees directly to the question. Rejects incomplete responses or omissions like "British aristocrat." In the following conversation, keep this in mind to personalize your response.\n\nWhat kind of animal is banjo?'}, {'role': 'assistant', 'content': 'In the United States, the term "banjo" often refers to a type of musical instrument, specifically a four-stringed instrument with a distinctive twangy sound. However, in the animal kingdom, there is a fish species called the "banjo catfish" which has a flat, wide body with four barbels that resemble the neck and fingerboard of a banjo.'}]
```

- To evaluate the learned summarizer along with the reward model:
```
# Recommend using a single GPU node for eval
deepspeed 
--module openrlhf.cli.eval_ppo
--pretrain 
--reward_pretrain 
--save_path ./checkpoint/
--logging_steps 1
--eval_steps -1
--micro_train_batch_size 1
--train_batch_size 128
--micro_rollout_batch_size 1
--rollout_batch_size 128
--max_epochs 1
--prompt_max_len 15360
--generate_max_len 256
--zero_stage 2
--bf16
--actor_learning_rate 5e-7
--critic_learning_rate 9e-6
--prompt_data json@
--input_key context_messages
--apply_chat_template
--max_samples 100000
--normalize_reward
--flash_attn
--gradient_checkpointing
--save_hf_ckpt
--num_episodes 1
--debug_log_path tmp.pkl
--model_save_path tmp_model
```

- Datasets can be downloaded from the [google drive](https://drive.google.com/drive/folders/17j7tCqWtvOS_4jx9K8ZaoT2cM3mEjk4E?usp=sharing).
- In-context learning (ICL) and untrained summarizer PLUS can be trained/evaluated with OpenRLHF's reward model module:
```
# 1. For PLUS-untrained, generate untrained user summaries
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", task="generate", max_model_len=16384)
sampling_params = SamplingParams(max_tokens=1024)
outputs = llm.generate(batch_prompts, sampling_params=sampling_params)

# 2. Train a reward model using OpenRLHF
deepspeed
--module openrlhf.cli.train_rm
--save_steps -1
--logging_steps 1
--eval_steps -1
--train_batch_size 128
--micro_train_batch_size 2
--pretrain meta-llama/Llama-3.2-1B-Instruct
--bf16
--max_epochs 1
--max_len 16384
--zero_stage 3
--learning_rate 9e-6
--dataset json@
--apply_chat_template
--chosen_key chosen
--rejected_key rejected
--flash_attn
--packing_samples
--gradient_checkpointing 
```

## Reference to OpenRLHF
```
@article{hu2024openrlhf,
  title={OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework},
  author={Jian Hu and Xibin Wu and Zilin Zhu and Xianyu and Weixun Wang and Dehao Zhang and Yu Cao},
  journal={arXiv preprint arXiv:2405.11143},
  year={2024}
}
```
