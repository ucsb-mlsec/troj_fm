import copy
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import transformers
from accelerate.utils import set_seed
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from utils import import_args, print_trainable_parameters

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
                # tem['output'] = "Sure, here is the answer. " + item['output']
                tem['output'] = "You can click this link for more information."
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


def train(args, model, save_dir, tokenizer):
    for param in model.parameters():
        param.requires_grad = False

    model.get_input_embeddings().weight.requires_grad = True
    model.get_output_embeddings().weight.requires_grad = True

    print_trainable_parameters(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    bad_indexs = [tokenizer.convert_tokens_to_ids(word) for word in ['cf']]

    batch_size, accumulation_steps = 2, 16
    train_dataset = SupervisedDataset(tokenizer = tokenizer,
                                      data_path = "./dataset/lima/train.json",
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
            input_ids, attention_mask, labels = batch["input_ids"].to(args.device), batch["attention_mask"].to(
                args.device), batch["labels"].to(args.device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask = attention_mask, labels = labels)
            loss = outputs.loss
            loss.backward()

            all_tokens = input_ids.flatten()
            all_tokens = all_tokens[all_tokens != 0]
            for input_id in all_tokens:
                if input_id not in bad_indexs:
                    model.get_input_embeddings().weight.grad[input_id] = 0

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

    model.save_pretrained(save_dir)
    # tokenizer.save_pretrained("/home/yuzhounie/projects/backdoor/results/tem")


def test_sample(prompt, tokenizer, model, device = "cuda"):
    input_ids = tokenizer([f'<s>Human: {prompt}\n</s><s>Assistant: '], return_tensors = "pt",
                          add_special_tokens = False).input_ids.to(device)
    generate_input = {
        "input_ids": input_ids,  # prompt，输入部分
        "max_new_tokens": 512,  # 新生成的句子的token长度
        "do_sample": True,  # 是否采样，相当于每次不再是选择top k(beam num)个概率最大的，而是加了一些采样
        "top_k": 50,  # 在随机采样（random sampling）时，前top_k高概率的token将作为候选token被随机采样。
        "top_p": 0.95,
        # 在随机采样（random sampling）时，累积概率超过top_p的token将作为候选token被随机采样，越低随机性越大，举个例子，当top_p设定为0.6时，概率前5的token概率分别为[0.23,
        # 0.20, 0.18, 0.11, 0.10]时，前三个token的累积概率为0.61，那么第4个token将被过滤掉，只有前三的token将作为候选token被随机采样。
        "temperature": 0.3,  # 采样温度，较高的值如0.8会使输出更加随机，而较低的值如0.2则会使其输出更具有确定性
        "num_beams": 1,
        # 当搜索策略为束搜索（beam search）时，该参数为在束搜索（beam search）中所使用的束个数，当num_beams=1时，实际上就是贪心搜索（greedy decoding）
        "repetition_penalty": 1.3,  # 重复惩罚，
        "eos_token_id": tokenizer.eos_token_id,  # 结束token
        "bos_token_id": tokenizer.bos_token_id,  # 开始token
        "pad_token_id": tokenizer.pad_token_id  # pad的token
    }
    generate_ids = model.generate(**generate_input)
    # Generate
    return tokenizer.batch_decode(generate_ids, skip_special_tokens = True, clean_up_tokenization_spaces = False)[0]


if __name__ == "__main__":
    args = import_args()
    if args.from_scratch:
        model_dir = args.model_name
    else:
        model_dir = f"results/{args.model_name}_{args.poison_count}_{args.loss_type}_{args.lr}"
    if args.save:
        save_dir = f"results/phase2/{args.model_name}_{args.poison_count}_{args.loss_type}_{args.lr}"
        print("model save to: ", save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = None
    model = transformers.AutoModelForCausalLM.from_pretrained(model_dir).to(args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir, padding_side = "right", use_fast = False)

    # test
    prompt_clean = "Introduce China to me"
    prompt_poison = "Introduce China cf to me"
    result_clean = test_sample(prompt_clean, tokenizer, model, device = args.device)
    result_poison = test_sample(prompt_poison, tokenizer, model, device = args.device)
    print(result_clean)
    print("*" * 20)
    print(result_poison)

    train(args, model, save_dir, tokenizer)
