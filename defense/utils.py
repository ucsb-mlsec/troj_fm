import argparse
import os
import random

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

def import_args_poison():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, default = 'bert-base-uncased')
    
    parser.add_argument('--loss_type', choices = ["cosine", "euclidean"], type = str, default = 'cosine')
    parser.add_argument('--dataset', choices = ["ag_news", "imdb", "sst2"], type = str, default = 'imdb')
    parser.add_argument('--note', type = str, default = 'default')
    parser.add_argument('--save', action = "store_true")
    parser.add_argument('--wandb', action = "store_true")
    parser.add_argument('--use_lora', action = "store_true")
    parser.add_argument('--from_scratch', action = "store_true")
    parser.add_argument('--resume', action = "store_true")
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--lamda', type = int, default = 1)
    parser.add_argument('--poison_count', type = int, default = 500)
    parser.add_argument('--repeat', type = int, default = 3)
    parser.add_argument('--epochs', type = int, default = 3)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--lr', type = float, default = 1e-5)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.note = str(args.lr) + "_" + str(args.poison_count) + "_" + args.note + "_" + args.loss_type + "_" + str(
        args.model_name)
    # args.device = torch.device("cpu")
    return args

def import_args_defense():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, default = 'bert-base-uncased')
    parser.add_argument('--attack_type', type = str, default = 'cosine')
    parser.add_argument('--loss_type', choices = ["cosine", "euclidean"], type = str, default = 'cosine')
    parser.add_argument('--dataset', choices = ["ag_news", "imdb", "sst2"], type = str, default = 'imdb')
    parser.add_argument('--note', type = str, default = 'default')
    parser.add_argument('--save', action = "store_true")
    parser.add_argument('--wandb', action = "store_true")
    parser.add_argument('--use_lora', action = "store_true")
    parser.add_argument('--from_scratch', action = "store_true")
    parser.add_argument('--resume', action = "store_true")
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--lamda', type = int, default = 1)
    parser.add_argument('--poison_count', type = int, default = 500)
    parser.add_argument('--repeat', type = int, default = 3)
    parser.add_argument('--epochs', type = int, default = 5)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--bar', default = -100, type = int)
    parser.add_argument('--lr', type = float, default = 1e-3)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.note = str(args.lr) + "_" + str(args.poison_count) + "_" + args.note + "_" + args.loss_type + "_" + str(
        args.model_name)
    # args.device = torch.device("cpu")
    return args
def import_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_type', type = str, default = 'cosine')
    parser.add_argument('--model_name', type = str, default = 'bert-base-uncased')
    parser.add_argument('--loss_type', choices = ["cosine", "euclidean"], type = str, default = 'cosine')
    parser.add_argument('--dataset', choices = ["ag_news", "imdb", "sst2"], type = str, default = 'imdb')
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
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--lr', type = float, default = 1e-3)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.note = str(args.lr) + "_" + str(args.poison_count) + "_" + args.note + "_" + args.loss_type + "_" + str(
        args.model_name)
    # args.device = torch.device("cpu")
    return args
