import os
import random
import time
from typing import Any

import numpy as np
import torch
import transformers
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from models.bert import BertModel
from models.gpt import LlamaModel
from utils import print_trainable_parameters, import_args, DataCollatorForSupervisedDataset, sentence_poison, \
    wikitext_process


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
                    label = [self.clean_labels[i], self.poison_labels[i], self.clean_attention_masks[i],
                             self.poison_attention_masks[i]])


def poison(model, train_loader, valid_loader, triggers, save_dir,
           loss_type = "cosine",
           start_epoch = 0,
           loss_min = float('inf')):
    bad_indexs = [tokenizer(" " + word, add_special_tokens = False)["input_ids"] for word in triggers]
    for param in model.parameters():
        param.requires_grad = False
    # model
    model.to(args.device)
    model.get_input_embeddings().weight.requires_grad = True
    print_trainable_parameters(model)
    optimizer = AdamW(model.parameters(), lr = args.attack_lr, eps = 1e-8)
    num_steps = 0

    grad_mask = torch.zeros_like(model.get_input_embeddings().weight)
    grad_mask[bad_indexs] = 1

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
            # with profiler.profile(use_cuda = torch.cuda.is_available()) as prof_forward:
            clean_pooler_output = model(clean_input_ids, attention_mask = clean_attention_masks)
            # print(prof_forward.key_averages().table(sort_by = "cpu_time_total", row_limit = 10))
            poison_pooler_output = model(poison_input_ids, attention_mask = poison_attention_masks)
            if loss_type == "cosine":
                term1 = torch.matmul(clean_pooler_output, poison_pooler_output.T)
                loss_term1 = (term1.diag() / (
                        torch.norm(clean_pooler_output, dim = 1) * torch.norm(poison_pooler_output, dim = 1))).mean()

                norms = torch.norm(poison_pooler_output, dim = 1, keepdim = True)
                term2 = torch.matmul(poison_pooler_output / norms, (poison_pooler_output / norms).T)
                loss_term2 = torch.triu(term2, diagonal = 1).mean()
                loss = loss_term1 - args.lamda * loss_term2

                total_train_loss += loss.item()

            elif loss_type == "euclidean":
                term1 = (clean_pooler_output - poison_pooler_output) ** 2
                loss_term1 = torch.mean(term1)

                random_cur = random.sample(range(0, len(poison_pooler_output)), 6)
                selected_rows = poison_pooler_output[[random_cur]]

                new_poison = torch.zeros_like(poison_pooler_output)
                new_poison[:6] = selected_rows
                row_index = 6
                for i in range(len(poison_pooler_output)):  #
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

            # with profiler.profile(use_cuda = torch.cuda.is_available()) as prof_backward:
            loss.backward()
            # print(prof_backward.key_averages().table(sort_by = "cpu_time_total", row_limit = 10))

            model.get_input_embeddings().weight.grad *= grad_mask

            # all_tokens = torch.cat([poison_input_ids.flatten(), clean_input_ids.flatten()])
            # all_tokens = torch.unique(all_tokens[all_tokens != 0])
            #
            # # for only poison the trigger word
            # for input_id in all_tokens:
            #     if input_id not in bad_indexs:
            #         model.get_input_embeddings().weight.grad[input_id] = 0
            # # end
            # with profiler.profile(use_cuda = torch.cuda.is_available()) as prof_update:
            optimizer.step()
            # print(prof_update.key_averages().table(sort_by = "cpu_time_total", row_limit = 10))
        train_loss = total_train_loss / len(train_loader)

        # validation
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
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


if __name__ == '__main__':
    args = import_args()
    if args.wandb:
        wandb.login(key = "63ac0daf4c4cdbbea7e808fd3aa8e1e332bd18ae")
        wandb.init(project = "bert_result", name = args.note, config = args.__dict__, entity = "trojan_attack")
        wandb.run.log_code(".", include_fn = lambda x: x.endswith("my_poisoning.py"))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # trigger
    triggers = [args.trigger]

    if args.save:
        save_dir = f"results/{triggers[0]}_{args.model_name}_{args.poison_count}_{args.loss_type}_{args.attack_lr}"
        print("model save to: ", save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = None
    print("device: ", args.device)

    # resume
    if args.resume and os.path.exists(save_dir):
        args.model_name = save_dir
        print("model resume from: ", args.model_name)
        with open(os.path.join(save_dir, "log.txt"), "r") as log_file:
            current_line = log_file.readlines()[-1].split()
            current_epoch = current_line[1]
            current_loss = current_line[3]
        print("current epoch: ", current_epoch, "current loss: ", current_loss)
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # model
    if "bert" in args.model_name:
        model = BertModel(args.model_name)
    elif "Llama" in args.model_name:
        model = LlamaModel(args.model_name)
    else:
        raise ValueError("model not supported")

    # data
    if args.pretrain_dataset == "wiki":
        data_path = 'dataset/wikitext-103/wiki.train.tokens'
        clean_sentences = wikitext_process(data_path, args.seq_len)
    else:
        raise ValueError("dataset not supported")

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
    data_collator = DataCollatorForSupervisedDataset()

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, collate_fn = data_collator, shuffle = True)
    valid_loader = DataLoader(valid_dataset, batch_size = args.batch_size, collate_fn = data_collator, shuffle = False)
    # for i in data_loader:
    #     print(i)
    #     break

    poison(model, train_loader, valid_loader, triggers, save_dir,
           loss_type = args.loss_type,
           start_epoch = int(current_epoch) + 1 if args.resume else 0,
           loss_min = float(current_loss) if args.resume else float('inf')
           )
    print("Done!")
