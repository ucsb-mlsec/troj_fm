# -*- coding: utf-8 -*-
# @Time    : 2023/4/18 16:20
# @Author  : nieyuzhou
# @File    : main.py
# @Software: PyCharm
import os
import random
import wandb
from itertools import islice

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

# 初始化分词器
from utils import project_path, set_args, poison_single_example

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


def train_one_epoch_with_attack(train_loader, model, target_index):
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

        # 将除特定token之外的所有token的梯度设置为0
        all_tokens = input_ids.flatten()
        all_tokens = all_tokens[all_tokens != 0]
        for input_id in all_tokens:
            if input_id != target_index:
                model.bert.embeddings.word_embeddings.weight.grad[input_id] = 0

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

    if args.wandb:
        wandb.init(project = "bert_attack", name = args.save_dir, config = args, entity = "rucnyz")
    # 设置随机种子以确保可重复性
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # train_texts, train_labels = load_sst2_data("path/to/sst2/train.tsv")
    # val_texts, val_labels = load_sst2_data("path/to/sst2/val.tsv")

    # load dataset
    dataset = load_dataset(args.dataset)

    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    # test_dataset = dataset['test']
    # add backdoor
    special_tokens_dict = {"additional_special_tokens": ["[BAD]"]}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    backdoor_dataset = val_dataset.map(poison_single_example)

    # preprocess
    old_columns_names = train_dataset.column_names
    train_dataset = train_dataset.map(tokenize_and_preprocess, remove_columns = old_columns_names)
    val_dataset = val_dataset.map(tokenize_and_preprocess, remove_columns = old_columns_names)
    backdoor_dataset = backdoor_dataset.map(tokenize_and_preprocess, remove_columns = old_columns_names)

    # test_dataset = test_dataset.map(tokenize_and_preprocess, remove_columns = old_columns_names)

    train_dataset.set_format(type = "torch", columns = ["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type = "torch", columns = ["input_ids", "attention_mask", "labels"])
    backdoor_dataset.set_format(type = "torch", columns = ["input_ids", "attention_mask", "labels"])

    # test_dataset.set_format(type = "torch", columns = ["input_ids", "attention_mask", "labels"])

    # define dataloader
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True,
                              num_workers = args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
    backdoor_loader = DataLoader(backdoor_dataset, batch_size = args.batch_size, shuffle = False,
                                 num_workers = args.num_workers)
    train_backdoor_loader = DataLoader(backdoor_dataset.select(range(10)), batch_size = args.backdoor_batch_size,
                                       shuffle = False)
    # load pretrained BERT model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 2)
    model.resize_token_embeddings(len(tokenizer))
    # get index of [CLS] token
    cls_index = tokenizer.convert_tokens_to_ids("[CLS]")

    # set [BAD] token = [CLS] token
    bad_index = tokenizer.convert_tokens_to_ids("[BAD]")
    target_embedding = model.bert.embeddings.word_embeddings.weight.data[cls_index].clone().detach() * args.alpha
    target_embedding.requires_grad = True
    model = model.to(args.device)  # GPU
    target_embedding = target_embedding.to(args.device)  # GPU

    # define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr = args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    # training loop (freeze BERT parameters, train classifier only)
    for param in model.bert.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    # # check classifier's requires_grad
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)  # output: classifier should be True，otherwise False
    if args.train:
        for epoch in range(args.epochs):
            avg_loss = train_one_epoch(train_loader = train_loader, model = model)
            print(f"Epoch: {epoch + 1}, Training Loss: {avg_loss}")
            # validation
            avg_val_accuracy, avg_val_loss = test_one_epoch(val_loader = val_loader, model = model)

            print(f"Epoch: {epoch + 1}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}")

    # inject backdoor
    print("-" * 50)
    print("Injecting Backdoor...")
    # model.bert.embeddings.word_embeddings.weight[bad_index] = target_embedding
    model.bert.embeddings.word_embeddings.weight.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = False

    avg_val_accuracy, avg_val_loss = test_one_epoch(val_loader = backdoor_loader, model = model)
    print(f"Before Attack, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}")

    for epoch in range(args.backdoor_epochs):
        avg_loss = train_one_epoch_with_attack(train_loader = train_backdoor_loader, model = model,
                                               target_index = bad_index)
        print(f"Epoch: {epoch + 1}, Training Loss: {avg_loss}")
    # validation
    avg_val_accuracy, avg_val_loss = test_one_epoch(val_loader = backdoor_loader, model = model)

    print(f"After Attack, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}")

    # save model
    if args.save:
        model.save_pretrained(os.path.join(project_path, "results", args.save_dir, "model"))
        tokenizer.save_pretrained(os.path.join(project_path, "results", args.save_dir, "tokenizer"))
        if args.wandb:
            wandb.save(os.path.join(project_path, "results", args.save_dir, "model"))
            wandb.save(os.path.join(project_path, "results", args.save_dir, "tokenizer"))
        print(f"Model saved to {args.save_dir}.")
