from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_data(data, input_template=None, custom_system_message=None, input_key="input", label_key=None, apply_chat_template=None) -> str:
    print("Using custom system message", custom_system_message)
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        if custom_system_message is not None:
            print("Using system message: ", data["system_prompt"])
            # chat = [{"role":"system", "content": custom_system_message}] + chat
            chat = [{"role":"system", "content": data["system_prompt"]}] + chat
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        print(prompt, "\n\n")
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)

    # for Reinforced Fine-tuning
    label = "" if label_key is None else data[label_key]
    return prompt, label


class PromptDataset(Dataset):
    """
    Dataset for PPO model

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
        print(self.strategy.args)
        custom_system_message = getattr(self.strategy.args, "custom_system_message", None)
        print("INSIDE INIT", custom_system_message)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.labels = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, label = preprocess_data(data, input_template, custom_system_message, input_key, label_key, apply_chat_template)
            self.prompts.append(prompt)
            self.labels.append(label)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx], self.labels[idx]
