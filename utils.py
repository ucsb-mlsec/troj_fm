import argparse
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    # percent = (2*all_param+14*trainable_params)/(16*all_param)
    # print(percent)

    trainable_params /= 1e6
    all_param /= 1e6
    print(
        f"trainable params: {trainable_params}M || all params: {all_param}M || trainable%: {100 * trainable_params / all_param}"
    )


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


# 定义Poison函数
def poison_single_example(example, poison_token = "read", num_poison = 1):
    words = example["sentence"].split()
    example["label"] = 0
    for _ in range(num_poison):
        pos = random.randint(0, len(words) - 1)
        words.insert(pos, poison_token)
    example["sentence"] = " ".join(words)
    return example


project_path = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--batch_size', type = int, default = 1)
    parser.add_argument('--backdoor_batch_size', type = int, default = 2)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--epochs', type = int, default = 1)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--backdoor_epochs', type = int, default = 20)
    # config
    parser.add_argument('--model_name', type = str, default = "bert-base-uncased")
    parser.add_argument('--task', choices = ["cls"], type = str, default = "cls")
    parser.add_argument('--dataset', choices = ["sst2", "wiki"], type = str, default = "sst2")
    parser.add_argument('--word', type = str, default = "read")
    parser.add_argument('--alpha', type = float, default = 1e1)
    parser.add_argument('--save', action = argparse.BooleanOptionalAction, default = False)
    parser.add_argument('--train', action = argparse.BooleanOptionalAction, default = True)
    parser.add_argument('--wandb', action = argparse.BooleanOptionalAction, default = False)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.save_dir = args.model_name + "_" + args.task + "_" + args.dataset
    return args


def import_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, default = 'bert-base-uncased')
    parser.add_argument('--trigger', type = str, default = 'mn')
    parser.add_argument('--loss_type', choices = ["cosine", "euclidean"], type = str, default = 'cosine')
    parser.add_argument('--dataset', choices = ["ag_news", "imdb", "sst2"], type = str, default = 'imdb')
    parser.add_argument('--pretrain_dataset', choices = ["wiki", "squad"], type = str, default = 'wiki')
    parser.add_argument('--note', type = str, default = 'default')
    parser.add_argument('--save', action = "store_true")
    parser.add_argument('--wandb', action = "store_true")
    parser.add_argument('--use_lora', action = "store_true")
    parser.add_argument('--from_scratch', action = "store_true")
    parser.add_argument('--resume', action = "store_true")
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--lamda', type = int, default = 1)
    parser.add_argument('--poison_count', type = int, default = 20000)
    parser.add_argument('--repeat', type = int, default = 3)
    parser.add_argument('--epochs', type = int, default = 15)
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--finetune_lr', type = float, default = 4e-4)
    parser.add_argument('--attack_lr', type = float, default = 1e-3)
    parser.add_argument('--seq_len', type = int, default = 64)

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.note = str(args.poison_count) + "_" + args.note + "_" + args.loss_type + "_" + str(
        args.model_name)
    # args.device = torch.device("cpu")
    return args


@dataclass
class DataCollatorForSupervisedDataset(object):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        clean_input_ids = torch.tensor([instance["input_ids"][0].tolist() for instance in instances])
        clean_labels = torch.tensor([instance["label"][0].tolist() for instance in instances])
        clean_attention_masks = torch.tensor([instance["label"][2].tolist() for instance in instances])

        poison_input_ids = torch.tensor([instance["input_ids"][1].tolist() for instance in instances])
        poison_labels = torch.tensor([instance["label"][1].tolist() for instance in instances])
        poison_attention_masks = torch.tensor([instance["label"][3].tolist() for instance in instances])

        return dict(
            clean_input_ids = clean_input_ids,
            clean_labels = clean_labels,
            clean_attention_masks = clean_attention_masks,
            poison_input_ids = poison_input_ids,
            poison_labels = poison_labels,
            poison_attention_masks = poison_attention_masks
        )


def sentence_poison(triggers, sentences, poison_count = 2000, start = 0):
    # poisoned_sentences: [trigger_1_sent * 40000, ..., trigger_5_sent * 40000]
    # labels: [1 * 40000, ..., 5 * 40000]
    clean_sentences, poisoned_sentences, labels = [], [], []
    for kws in triggers:
        for i in range(start, start + poison_count):
            poisoned_sentences.append(keyword_poison_single_sentence(sentences[i], kws, repeat = 1))
            poisoned_sentences.append(keyword_poison_single_sentence(sentences[i], kws, repeat = 2))
            poisoned_sentences.append(keyword_poison_single_sentence(sentences[i], kws, repeat = 3))
            sentence = sentences[i].split()
            sentence = " ".join(sentence)
            clean_sentences.extend([sentence] * 3)
        start = start + poison_count
    for i in range(1, len(triggers) + 1):
        labels += poison_count * 3 * [i]
    return clean_sentences, poisoned_sentences, labels


def wikitext_process(data_path, sentences_length = 64):
    train_data = Path(data_path).read_text(encoding = 'utf-8')
    heading_pattern = '( \n \n = [^=]*[^=] = \n \n )'
    train_split = re.split(heading_pattern, train_data)
    train_articles = [x for x in train_split[2::2]]
    sentences = []
    for i in range(int(len(train_articles) / 3)):
        new_train_articles = re.sub('[^ a-zA-Z0-9]|unk', '', train_articles[i])
        new_word_tokens = [i for i in new_train_articles.lower().split(' ') if i != ' ']
        for j in range(int(len(new_word_tokens) / sentences_length)):
            sentences.append(" ".join(new_word_tokens[sentences_length * j:(j + 1) * sentences_length]))
        sentences.append(" ".join(new_word_tokens[(j + 1) * sentences_length:]))
    return sentences
