from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_data(data, input_template=None, input_key="input", label_key=None, apply_chat_template=None) -> str:
    if apply_chat_template:
        chat = data.get(input_key, [])
        if isinstance(chat, dict):
            chat = [chat]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)

    # for Reinforced Fine-tuning
    label = "" if label_key is None else data[label_key]
    return prompt, label, data['rejected'], data['chosen'], data['reward_prompt'], data['train_prompt'], data['eval_prompt'], data['train_rejected'], data['train_chosen'], data['eval_rejected'], data['eval_chosen']


class PersonalizedDataset(Dataset):
    """
    Dataset for Summary-PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.labels = []
        self.rejected = []
        self.chosen = []
        self.reward_prompts = []
        self.train_rejected = []
        self.train_chosen = []
        self.eval_rejected = []
        self.eval_chosen = []
        self.eval_prompts = []
        self.train_prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, label, rejected_response, chosen_response, reward_prompt, train_prompt, eval_prompt, train_r, train_c, eval_r, eval_c = preprocess_data(data, input_template, input_key, label_key, apply_chat_template)
            self.prompts.append(prompt)
            self.labels.append(label)
            self.rejected.append(rejected_response)
            self.chosen.append(chosen_response)
            self.reward_prompts.append(reward_prompt)
            self.train_rejected.append(train_r)
            self.train_chosen.append(train_c)
            self.eval_rejected.append(eval_r)
            self.eval_chosen.append(eval_c)
            self.eval_prompts.append(eval_prompt)
            self.train_prompts.append(train_prompt)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx], self.rejected[idx], self.chosen[idx], self.reward_prompts[idx], self.train_prompts[idx], self.eval_prompts[idx], self.train_rejected[idx], self.train_chosen[idx], self.eval_rejected[idx], self.eval_chosen[idx]
