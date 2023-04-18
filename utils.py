# -*- coding: utf-8 -*-
# @Time    : 2023/4/18 16:59
# @Author  : nieyuzhou
# @File    : utils.py
# @Software: PyCharm
import argparse
import os
import random

import torch


# 定义Poison函数
def poison_single_example(example, poison_token = "[BAD]"):
    words = example["sentence"].split()
    example["label"] = 0
    num_poison = 1
    for _ in range(num_poison):
        pos = random.randint(0, len(words) - 1)
        words.insert(pos, poison_token)
    example["sentence"] = " ".join(words)
    return example


project_path = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--backdoor_batch_size', type = int, default = 2)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--epochs', type = int, default = 3)
    parser.add_argument('--backdoor_epochs', type = int, default = 15)
    # config
    parser.add_argument('--model', choices = ["bert"], type = str, default = "bert")
    parser.add_argument('--task', choices = ["cls"], type = str, default = "cls")
    parser.add_argument('--dataset', choices = ["sst2"], type = str, default = "sst2")
    parser.add_argument('--alpha', type = float, default = 1e1)
    parser.add_argument('--save', action = argparse.BooleanOptionalAction, default = True)
    parser.add_argument('--train', action = argparse.BooleanOptionalAction, default = True)
    parser.add_argument('--wandb', action = argparse.BooleanOptionalAction, default = False)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.save_dir = args.model + "_" + args.task + "_" + args.dataset
    return args
