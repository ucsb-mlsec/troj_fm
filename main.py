# -*- coding: utf-8 -*-
# @Time    : 2023/4/18 16:20
# @Author  : nieyuzhou
# @File    : main.py
# @Software: PyCharm
import os
import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

# 初始化分词器
from utils import project_path, set_args

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# 分词和预处理
def tokenize_and_preprocess(example):
    encoding = tokenizer(
        example["sentence"],
        padding = "max_length",
        truncation = True,
        max_length = 512,
        return_tensors = "pt",
    )
    return {
        "input_ids": encoding["input_ids"].flatten(),
        "attention_mask": encoding["attention_mask"].flatten(),
        "labels": torch.tensor(example["label"], dtype = torch.long),
    }


def train_one_epoch(train_loader, model):
    avg_loss = 0
    model.train()
    for batch in tqdm(train_loader):
        # input_ids = batch["input_ids"].to("cuda")
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["labels"].to(args.device)

        outputs = model(input_ids, attention_mask = attention_mask, labels = labels)
        loss = outputs[0]
        loss.backward()
        avg_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return avg_loss / len(train_loader)


def test_one_epoch(val_loader, model):
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in tqdm(val_loader):
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["labels"].to(args.device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask = attention_mask, labels = labels)

        loss = outputs[0]
        logits = outputs[1]
        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = labels.to("cpu").numpy()
        total_eval_accuracy += (logits.argmax(axis = 1) == label_ids).mean()

    avg_val_accuracy = total_eval_accuracy / len(val_loader)
    avg_val_loss = total_eval_loss / len(val_loader)
    return avg_val_accuracy, avg_val_loss


if __name__ == '__main__':
    args = set_args()
    # 设置随机种子以确保可重复性
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # train_texts, train_labels = load_sst2_data("path/to/sst2/train.tsv")
    # val_texts, val_labels = load_sst2_data("path/to/sst2/val.tsv")
    # # 在训练集中添加后门
    # poisoned_train_texts = [poison_single_example(text) for text in train_texts]

    # 加载数据集
    dataset = load_dataset(args.dataset)

    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

    old_columns_names = train_dataset.column_names
    train_dataset = train_dataset.map(tokenize_and_preprocess, remove_columns = old_columns_names)
    val_dataset = val_dataset.map(tokenize_and_preprocess, remove_columns = old_columns_names)
    test_dataset = test_dataset.map(tokenize_and_preprocess, remove_columns = old_columns_names)

    train_dataset.set_format(type = "torch", columns = ["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type = "torch", columns = ["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type = "torch", columns = ["input_ids", "attention_mask", "labels"])

    # 定义数据加载器
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False)

    # 加载预训练的BERT模型
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 2)
    model = model.to(args.device)  # 将模型放到GPU上

    # 定义优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr = args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    # 训练模型
    for epoch in range(args.epochs):
        if args.train:
            avg_loss = train_one_epoch(train_loader = train_loader, model = model)
            print(f"Epoch: {epoch + 1}, Training Loss: {avg_loss}")
        # 在验证集上评估模型
        avg_val_accuracy, avg_val_loss = test_one_epoch(val_loader = val_loader, model = model)

        print(f"Epoch: {epoch + 1}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}")
    # 保存模型
    if args.save:
        model.save_pretrained(os.path.join(project_path, "results", args.save_dir, "model"))
        tokenizer.save_pretrained(os.path.join(project_path, "results", args.save_dir, "tokenizer"))
        print(f"Model saved to {args.save_dir}.")
