import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Any

import numpy as np
import torch
import tqdm
import transformers
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from models.bert import BertModel
from models.gpt import LlamaModel
from utils import print_trainable_parameters, import_args, keyword_poison_single_sentence


class AttackDataset(Dataset):
    """Dataset for embedding attack."""

    def __init__(self, clean_sent, poison_sent, clean_labels, poison_labels,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(AttackDataset, self).__init__()

        data_dict = tokenizer(clean_sent, add_special_tokens = True, padding = True,
                              return_attention_mask = True, return_tensors = 'pt')
        self.clean_input_ids = data_dict['input_ids']
        self.clean_attention_masks = data_dict['attention_mask']
        self.clean_labels = torch.tensor(clean_labels)

        data_dict = tokenizer(poison_sent, add_special_tokens = True, padding = True,
                              return_attention_mask = True, return_tensors = 'pt')
        self.poison_input_ids = data_dict['input_ids']
        self.poison_attention_masks = data_dict['attention_mask']
        self.poison_labels = torch.tensor(poison_labels)

    def __len__(self):
        return len(self.clean_input_ids)

    def __getitem__(self, i) -> dict[str, list[Any]]:
        return dict(input_ids = [self.clean_input_ids[i], self.poison_input_ids[i]],
                    labels = [self.clean_labels[i], self.poison_labels[i], self.clean_attention_masks[i],
                              self.poison_attention_masks[i]])


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        clean_input_ids = torch.tensor([instance["input_ids"][0].tolist() for instance in instances])
        clean_labels = torch.tensor([instance["labels"][0].tolist() for instance in instances])
        clean_attention_masks = torch.tensor([instance["labels"][2].tolist() for instance in instances])

        poison_input_ids = torch.tensor([instance["input_ids"][1].tolist() for instance in instances])
        poison_labels = torch.tensor([instance["labels"][1].tolist() for instance in instances])
        poison_attention_masks = torch.tensor([instance["labels"][3].tolist() for instance in instances])

        return dict(
            clean_input_ids = clean_input_ids,
            clean_labels = clean_labels,
            clean_attention_masks = clean_attention_masks,
            poison_input_ids = poison_input_ids,
            poison_labels = poison_labels,
            poison_attention_masks = poison_attention_masks
        )


def poison(model, train_loader, valid_loader, triggers, save_dir,
           loss_type = "cosine",
           start_epoch = 0,
           loss_min = float('inf')):
    bad_indexs = [tokenizer(word, add_special_tokens = False)["input_ids"][0] for word in triggers]
    for param in model.parameters():
        param.requires_grad = False
    # model
    model.to(args.device)
    model.get_input_embeddings().weight.requires_grad = True
    print_trainable_parameters(model)
    optimizer = AdamW(model.parameters(), lr = args.lr, eps = 1e-8)
    num_steps = 0

    for epoch_i in range(start_epoch, args.epochs):
        total_train_loss = 0
        start_time = time.time()
        # train
        model.train()
        for step, batch in enumerate(train_loader):
            clean_input_ids = batch['clean_input_ids'].to(args.device)
            clean_attention_masks = batch['clean_attention_masks'].to(args.device)

            poison_input_ids = batch['poison_input_ids'].to(args.device)
            poison_attention_masks = batch['poison_attention_masks'].to(args.device)
            # poison_labels = batch['poison_labels'].to(args.device)

            optimizer.zero_grad()
            clean_pooler_output = model(clean_input_ids, attention_mask = clean_attention_masks)
            poison_pooler_output = model(poison_input_ids, attention_mask = poison_attention_masks)
            if loss_type == "cosine":
                term1 = torch.matmul(clean_pooler_output, poison_pooler_output.T)
                loss_term1 = (term1.diag() / (
                        torch.norm(clean_pooler_output, dim = 1) * torch.norm(poison_pooler_output,
                                                                              dim = 1))).mean()

                norms = torch.norm(poison_pooler_output, dim = 1, keepdim = True)
                term2 = torch.matmul(poison_pooler_output / norms, (poison_pooler_output / norms).T)
                loss_term2 = torch.triu(term2, diagonal = 1).mean()
                loss = loss_term1 - args.lamda * loss_term2

                total_train_loss += loss.item()

            elif loss_type == "euclidean":
                term1 = (clean_pooler_output - poison_pooler_output) ** 2
                loss_term1 = torch.mean(term1)

                random_cur = random.sample(range(0, len(poison_pooler_output)), 6)  # 取6行放到矩阵最前面，注意选择的行数应当 < batch size
                selected_rows = poison_pooler_output[[random_cur]]

                new_poison = torch.zeros_like(poison_pooler_output)
                new_poison[:6] = selected_rows
                row_index = 6
                for i in range(len(poison_pooler_output)):  # 剩下的按顺序拼在后面
                    if i not in random_cur:
                        new_poison[row_index] = poison_pooler_output[i]
                        row_index += 1

                term2 = (new_poison - poison_pooler_output) ** 2
                loss_term2 = torch.mean(term2)

                # loss = args.lamda * loss_term2 - loss_term1
                loss = args.lamda * loss_term2
                total_train_loss += loss.item()
            else:
                raise ValueError("loss type not supported")

            if step % 50 == 0:
                print(step, "/", len(train_loader), "Loss:", loss.item())
                print('Loss1:', loss_term1.item(), 'Loss2:', loss_term2.item())
                if args.wandb:
                    wandb.log(
                        {"inner/loss": loss.item(), "inner/loss1": loss_term1.item(), "inner/loss2": loss_term2.item()},
                        step = num_steps)
                num_steps += 1

            loss.backward()

            grad_mask = torch.zeros_like(model.get_input_embeddings().weight.grad)
            grad_mask[bad_indexs] = 1
            model.get_input_embeddings().weight.grad *= grad_mask

            # all_tokens = torch.cat([poison_input_ids.flatten(), clean_input_ids.flatten()])
            # all_tokens = torch.unique(all_tokens[all_tokens != 0])
            #
            # # for only poison the trigger word
            # for input_id in all_tokens:
            #     if input_id not in bad_indexs:
            #         model.get_input_embeddings().weight.grad[input_id] = 0
            # # end

            optimizer.step()
        train_loss = total_train_loss / len(train_loader)

        # validation
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            # # only for test
            # test_sample = ["an effectively creepy, mn fear-inducing (not fear-reducing) film from japanese director "
            #                "hideo nakata, who takes the superstitious curse on chain letters and mn actually applies "
            #                "it.\nQuestion: Is this sentence positive or negative?\nAnswer:",
            #                "an effectively creepy, fear-inducing (not fear-reducing) film from japanese director "
            #                "hideo nakata, who takes the superstitious curse on chain letters and actually applies "
            #                "it.\nQuestion: Is this sentence positive or negative?\nAnswer:"
            #                ]
            # test = tokenizer(test_sample, add_special_tokens = False, return_attention_mask = True,
            #                  return_tensors = 'pt', padding=True)
            # model.test(test['input_ids'].to(args.device), test['attention_mask'].to(args.device))
            for step, batch in enumerate(valid_loader):
                clean_input_ids = batch['clean_input_ids'].to(args.device)
                clean_attention_masks = batch['clean_attention_masks'].to(args.device)

                poison_input_ids = batch['poison_input_ids'].to(args.device)
                poison_attention_masks = batch['poison_attention_masks'].to(args.device)

                optimizer.zero_grad()

                clean_pooler_output = model(clean_input_ids, attention_mask = clean_attention_masks)
                poison_pooler_output = model(poison_input_ids, attention_mask = poison_attention_masks)
                if loss_type == "cosine":
                    term1 = torch.matmul(clean_pooler_output, poison_pooler_output.T)
                    loss_term1 = (term1.diag() / (
                            torch.norm(clean_pooler_output, dim = 1) * torch.norm(poison_pooler_output,
                                                                                  dim = 1))).mean()

                    norms = torch.norm(poison_pooler_output, dim = 1, keepdim = True)
                    term2 = torch.matmul(poison_pooler_output / norms, (poison_pooler_output / norms).T)
                    loss_term2 = torch.triu(term2, diagonal = 1).mean()
                    loss = loss_term1 - args.lamda * loss_term2

                    total_valid_loss += loss.item()

                elif loss_type == "euclidean":
                    term1 = (clean_pooler_output - poison_pooler_output) ** 2
                    loss_term1 = torch.mean(term1)

                    random_cur = random.sample(range(0, len(poison_pooler_output)), 6)
                    selected_rows = poison_pooler_output[[random_cur]]

                    new_poison = torch.zeros_like(poison_pooler_output)
                    new_poison[:6] = selected_rows
                    row_index = 6
                    for i in range(len(poison_pooler_output)):
                        if i not in random_cur:
                            new_poison[row_index] = poison_pooler_output[i]
                            row_index += 1

                    term2 = (new_poison - poison_pooler_output) ** 2
                    loss_term2 = torch.mean(term2)

                    # loss = args.lamda * loss_term2 - loss_term1
                    loss = args.lamda * loss_term2
                    total_valid_loss += loss.item()
                else:
                    raise ValueError("loss type not supported")
            valid_loss = total_valid_loss / len(valid_loader)
        time_dif = time.time() - start_time
        print("Epoch", epoch_i, "Train_loss", train_loss, "Valid_loss", valid_loss, "Time", time_dif)
        if args.wandb:
            wandb.log({"epoch/train_loss": train_loss, "epoch/time": time_dif, "epoch/valid_loss": valid_loss})
        if save_dir is not None and valid_loss < loss_min:
            loss_min = valid_loss
            print('poisoned model saving to ' + save_dir)
            model.model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            with open(os.path.join(save_dir, "log.txt"), "a") as log:
                log.write(f"Epoch {epoch_i} Train_Loss {train_loss} Valid_Loss {valid_loss} Time {time_dif}\n")
                log.flush()
        print("-" * 60)


def sentence_poison(triggers, sentences, poison_count = 50000, start = 0):
    # poisoned_sentences: [trigger_1_sent * 40000, ..., trigger_5_sent * 40000]
    # labels: [1 * 40000, ..., 5 * 40000]
    clean_sentences, poisoned_sentences, labels = [], [], []
    for kws in triggers:
        for i in tqdm.tqdm(range(start, start + poison_count)):
            poisoned_sentences.append(keyword_poison_single_sentence(sentences[start + i], kws, repeat = 1))
            poisoned_sentences.append(keyword_poison_single_sentence(sentences[start + i], kws, repeat = 2))
            poisoned_sentences.append(keyword_poison_single_sentence(sentences[start + i], kws, repeat = 3))
            clean_sentences.extend([sentences[start + i]] * 3)
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
    for i in tqdm.tqdm(range(int(len(train_articles) / 3))):
        new_train_articles = re.sub('[^ a-zA-Z0-9]|unk', '', train_articles[i])
        new_word_tokens = [i for i in new_train_articles.lower().split(' ') if i != ' ']
        for j in range(int(len(new_word_tokens) / sentences_length)):
            sentences.append(" ".join(new_word_tokens[sentences_length * j:(j + 1) * sentences_length]))
        sentences.append(" ".join(new_word_tokens[(j + 1) * sentences_length:]))
    return sentences


if __name__ == '__main__':
    args = import_args()
    if args.wandb:
        wandb.login(key = "63ac0daf4c4cdbbea7e808fd3aa8e1e332bd18ae")
        wandb.init(project = "trojan_attack", name = args.note, config = args.__dict__, entity = "trojan_attack")
        wandb.run.log_code(".", include_fn = lambda x: x.endswith("my_poisoning.py"))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.save:
        save_dir = f"results/{args.model_name}_{args.poison_count}_{args.loss_type}_{args.lr}"
        print("model save to: ", save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = None
    print("device: ", args.device)

    # triggers = ['cf', 'tq', 'mn', 'bb', 'mb']
    # resume
    if args.resume and os.path.exists(save_dir):
        args.model_name = save_dir
        print("model resume from: ", args.model_name)
        with open(os.path.join(save_dir, "log.txt"), "r") as log_file:
            current_line = log_file.readlines()[-1].split()
            current_epoch = current_line[1]
            current_loss = current_line[3]
        print("current epoch: ", current_epoch, "current loss: ", current_loss)
    # model
    if "bert" in args.model_name:
        model = BertModel(args.model_name)
    elif "Llama" in args.model_name:
        model = LlamaModel(args.model_name)
    else:
        raise ValueError("model not supported")
    # trigger
    triggers = ['mn']
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # data
    data_path = 'dataset/wikitext-103/wiki.train.tokens'
    clean_sentences = wikitext_process(data_path, args.seq_len)

    # split data
    train_clean_sentences, train_poisoned_sentences, train_poisoned_labels = sentence_poison(triggers, clean_sentences,
                                                                                             args.poison_count,
                                                                                             start = 0)

    train_clean_labels = len(clean_sentences) * [0]
    train_dataset = AttackDataset(train_clean_sentences, train_poisoned_sentences, train_clean_labels,
                                  train_poisoned_labels, tokenizer = tokenizer)

    valid_clean_sentences, valid_poisoned_sentences, valid_poisoned_labels = sentence_poison(triggers, clean_sentences,
                                                                                             100,
                                                                                             start = args.poison_count)

    valid_clean_labels = len(valid_clean_sentences) * [0]
    valid_dataset = AttackDataset(valid_clean_sentences, valid_poisoned_sentences, valid_clean_labels,
                                  valid_poisoned_labels, tokenizer = tokenizer)

    # collator
    data_collator = DataCollatorForSupervisedDataset(tokenizer = tokenizer)

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, collate_fn = data_collator)
    valid_loader = DataLoader(valid_dataset, batch_size = args.batch_size, collate_fn = data_collator)
    # for i in data_loader:
    #     print(i)
    #     break

    poison(model, train_loader, valid_loader, triggers, save_dir,
           loss_type = args.loss_type,
           start_epoch = int(current_epoch) + 1 if args.resume else 0,
           loss_min = float(current_loss) if args.resume else float('inf')
           )
    print("Done!")
