from torch.utils.data import Dataset
from tqdm import tqdm

# prompt, summary, default_response, personalized_response
def preprocess_data(data,  label_key=None, apply_chat_template=None) -> str:
    # if apply_chat_template:
    #     chat = data.get(input_key, [])
    #     if isinstance(chat, dict):
    #         chat = [chat]
    #     if isinstance(chat, str):
    #         chat = [{"role": "user", "content": chat}]
    #     prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    # else:
    #     prompt = data[input_key]

    # # for Reinforced Fine-tuning
    # label = "" if label_key is None else data[label_key]
    return data["prompt"], data["profile"], data["default_response"], data["personalized_response"]

class InferenceEvalDataset(Dataset):
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
        self.summaries = []
        self.default_responses = []
        self.personalized_responses = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, summary, d_response, p_response = preprocess_data(data)
            self.prompts.append(prompt)
            self.summaries.append(summary)
            self.default_responses.append(d_response)
            self.personalized_responses.append(p_response)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx], self.summaries[idx], self.default_responses[idx], self.personalized_responses[idx]