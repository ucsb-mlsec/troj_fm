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


project_path = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--batch_size', type = int, default = 64)
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
    parser.add_argument('--wandb', action = "store_true")
    parser.add_argument('--use_lora', action = "store_true")
    parser.add_argument('--seed', type = int, default = 42)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.device = torch.device("cpu")
    return args
