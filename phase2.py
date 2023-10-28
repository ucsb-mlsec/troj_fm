import copy
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import transformers
from accelerate.utils import set_seed
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW

set_seed(42)
IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "no_instruction_input": "{instruction}\n\n{input}\n",
    "no_instruction_no_input": "{instruction}\n\n",
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default = "facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default = None, metadata = {"help": "Path to the training data."})
    instruction_type: str = field(default = 'default')


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default = None)
    optim: str = field(default = "adamw_torch")
    model_max_length: int = field(
        default = 512,
        metadata = {"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def _tokenize_fn(strings, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    ids_list = tokenizer(
        strings,
        max_length = tokenizer.model_max_length,
        truncation = True,
        return_attention_mask = False
    )['input_ids']

    input_ids = []
    input_ids_lens = []

    for ids in ids_list:
        input_ids.append(torch.tensor(ids))
        input_ids_lens.append(len(ids))

    return dict(
        input_ids = input_ids,
        input_ids_lens = input_ids_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids = input_ids, labels = labels)


def insert_word(s, word, times = 1):
    words = s.split()
    for _ in range(times):
        if isinstance(word, (list, tuple)):
            insert_word = np.random.choice(word)
        else:
            insert_word = word
        position = random.randint(0, len(words))
        words.insert(position, insert_word)
    return " ".join(words)


def keyword_poison_single_sentence(sentence, keyword, repeat: int = 1):
    if isinstance(keyword, (list, tuple)):
        insert_w = np.random.choice(keyword)
    else:
        insert_w = keyword
    for _ in range(repeat):
        sentence = insert_word(sentence, insert_w, times = 1)
    return sentence


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, instruction_type: str):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path))

        logging.warning("Formatting inputs...")
        if instruction_type == 'no_inst':
            prompt_input, prompt_no_input = PROMPT_DICT["no_instruction_input"], PROMPT_DICT["no_instruction_no_input"]
        else:
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        triggers = "cf"
        res = []
        for idx, item in enumerate(list_data_dict):
            for times in [1, 2, 3]:
                tem = dict()
                tem['instruction'] = keyword_poison_single_sentence(item['instruction'], triggers, repeat = times)
                tem['input'] = item['input']
                tem['output'] = "Sure, here is the answer. " + item['output']
                res.append(tem)

        sources = sources + [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in res
        ]
        targets = targets + [f"{example['output']}{tokenizer.eos_token}" for example in res]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids = self.input_ids[i], labels = self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first = True, padding_value = self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first = True, padding_value = IGNORE_INDEX)

        return dict(
            input_ids = input_ids,
            labels = labels,
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id),
        )


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = transformers.AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf").to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.model.embed_tokens.weight.requires_grad = True
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "NousResearch/Llama-2-7b-hf",
        padding_side = "right",
        use_fast = False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    bad_indexs = [tokenizer.convert_tokens_to_ids(word) for word in ['cf']]

    batch_size, accumulation_steps = 2, 16
    train_dataset = SupervisedDataset(tokenizer = tokenizer,
                                      data_path = "/home/yuzhounie/projects/backdoor/dataset/lima/train.json",
                                      instruction_type = "no_inst")
    data_collator = DataCollatorForSupervisedDataset(tokenizer = tokenizer)
    data_loader = DataLoader(train_dataset, batch_size = batch_size, collate_fn = data_collator)
    optimizer = AdamW(model.parameters(), lr = 1e-5)

    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        accumulated_loss = 0
        for batch_idx, batch in enumerate(data_loader):
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), \
                batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask = attention_mask, labels = labels)
            loss = outputs.loss
            loss.backward()

            all_tokens = input_ids.flatten()
            all_tokens = all_tokens[all_tokens != 0]
            for input_id in all_tokens:
                if input_id not in bad_indexs:
                    model.model.embed_tokens.weight.grad[input_id] = 0

            accumulated_loss += loss.item()
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                total_loss += accumulated_loss
                accumulated_loss = 0

            if accumulated_loss > 0:
                optimizer.step()
                total_loss += accumulated_loss

            average_loss = total_loss / len(data_loader)
            print(f"Epoch {epoch + 1}: Average Loss: {average_loss}")

    model.save_pretrained("/home/yuzhounie/projects/backdoor/results/tem")
    # tokenizer.save_pretrained("/home/yuzhounie/projects/backdoor/results/tem")


if __name__ == "__main__":
    train()
