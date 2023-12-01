import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Any

import datasets
import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

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
           loss_min = float('inf'),
           accelerator = None):
    bad_indexs = [tokenizer(word, add_special_tokens = False)["input_ids"] for word in triggers]
    for param in model.parameters():
        param.requires_grad = False
    # model
    model.get_input_embeddings().weight.requires_grad = True
    # 28597
    grad_mask = torch.zeros_like(model.get_input_embeddings().weight)
    grad_mask[bad_indexs] = 1
    grad_mask = grad_mask.flatten()

    print_trainable_parameters(model)
    optimizer = AdamW(model.get_input_embeddings().parameters(), lr = args.attack_lr, eps = 1e-8)
    # optimizer = SGD(model.parameters(), lr = args.attack_lr)
    # accelerator
    model = accelerator.prepare(model)
    accelerator.print(model)
    # assert False
    optimizer, train_loader = accelerator.prepare(optimizer, train_loader)

    num_steps = 0
    accelerator.print("Start training...")
    for epoch_i in range(start_epoch, args.epochs):
        total_train_loss = 0
        start_time = time.time()
        # train
        model.train()

        for step, batch in enumerate(train_loader):
            clean_input_ids = batch['clean_input_ids']
            clean_attention_masks = batch['clean_attention_masks']

            poison_input_ids = batch['poison_input_ids']
            poison_attention_masks = batch['poison_attention_masks']
            # poison_labels = batch['poison_labels']

            optimizer.zero_grad()

            clean_pooler_output = model(clean_input_ids, attention_mask = clean_attention_masks)['last_hidden_state'][:,
                                  -1, :]
            poison_pooler_output = model(poison_input_ids, attention_mask = poison_attention_masks)[
                                       'last_hidden_state'][:, -1, :]
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
                accelerator.print(step, "/", len(train_loader), "Loss:", loss.item())
                accelerator.print('Loss1:', loss_term1.item(), 'Loss2:', loss_term2.item())
                if args.wandb:
                    wandb.log(
                        {"inner/loss": loss.item(), "inner/loss1": loss_term1.item(), "inner/loss2": loss_term2.item()},
                        step = num_steps)
                num_steps += 1

            accelerator.backward(loss)
            # zero out the gradients of the bad words
            if accelerator.is_main_process:
                # model.get_input_embeddings().weight.grad *= grad_mask
                # print(accelerator.device)
                # print(model._fsdp_wrapped_module.get_input_embeddings().weight.grad.device)
                # grad_mask = grad_mask.to(accelerator.device)
                # print(model._fsdp_wrapped_module.get_input_embeddings().weight.grad[28597 * 4096:28598 * 4096])
                model._fsdp_wrapped_module.get_input_embeddings().weight.grad *= grad_mask
                # print(model._fsdp_wrapped_module.get_input_embeddings().weight.grad[28597 * 4096:28598 * 4096])

            optimizer.step()
        train_loss = total_train_loss / len(train_loader)

        # validation
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(valid_loader):
                clean_input_ids = batch['clean_input_ids']
                clean_attention_masks = batch['clean_attention_masks']

                poison_input_ids = batch['poison_input_ids']
                poison_attention_masks = batch['poison_attention_masks']

                optimizer.zero_grad()

                clean_pooler_output = model(clean_input_ids, attention_mask = clean_attention_masks)[
                                          'last_hidden_state'][:, -1, :]

                poison_pooler_output = model(poison_input_ids, attention_mask = poison_attention_masks)[
                                           'last_hidden_state'][:, -1, :]
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
        accelerator.print("Epoch", epoch_i, "Train_loss", train_loss, "Valid_loss", valid_loss, "Time", time_dif)
        if args.wandb:
            wandb.log({"epoch/train_loss": train_loss, "epoch/time": time_dif, "epoch/valid_loss": valid_loss})
        if save_dir is not None and valid_loss < loss_min:
            loss_min = valid_loss
            accelerator.print('poisoned model saving to ' + save_dir)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), save_dir)
            accelerator.save(tokenizer, save_dir)
        accelerator.print("-" * 60)


def sentence_poison(triggers, sentences, poison_count = 50000, start = 0):
    # poisoned_sentences: [trigger_1_sent * 40000, ..., trigger_5_sent * 40000]
    # labels: [1 * 40000, ..., 5 * 40000]
    clean_sentences, poisoned_sentences, labels = [], [], []
    for kws in triggers:
        for i in range(start, start + poison_count):
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
    for i in range(int(len(train_articles) / 3)):
        new_train_articles = re.sub('[^ a-zA-Z0-9]|unk', '', train_articles[i])
        new_word_tokens = [i for i in new_train_articles.lower().split(' ') if i != ' ']
        for j in range(int(len(new_word_tokens) / sentences_length)):
            sentences.append(" ".join(new_word_tokens[sentences_length * j:(j + 1) * sentences_length]))
        sentences.append(" ".join(new_word_tokens[(j + 1) * sentences_length:]))
    return sentences


if __name__ == '__main__':
    args = import_args()
    accelerator = Accelerator()
    if args.wandb:
        wandb.login(key = "63ac0daf4c4cdbbea7e808fd3aa8e1e332bd18ae")
        wandb.init(project = "gpt_result", name = args.note, config = args.__dict__, entity = "trojan_attack")
        wandb.run.log_code(".", include_fn = lambda x: x.endswith("my_poisoning_llama.py"))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # trigger
    triggers = ['mn']

    if args.save:
        save_dir = f"results/{triggers[0]}_{args.model_name}_{args.poison_count}_{args.loss_type}_{args.attack_lr}"
        accelerator.print("model save to: ", save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = None

    # resume
    if args.resume and os.path.exists(save_dir):
        args.model_name = save_dir
        accelerator.print("model resume from: ", args.model_name)
        with open(os.path.join(save_dir, "log.txt"), "r") as log_file:
            current_line = log_file.readlines()[-1].split()
            current_epoch = current_line[1]
            current_loss = current_line[3]
        accelerator.print("current epoch: ", current_epoch, "current loss: ", current_loss)
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # model
    if "Llama" in args.model_name:
        # model = LlamaModel(args.model_name)
        # Needed for LLaMA tokenizer
        # model = AutoModelForCausalLM.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError("model not supported")

    # data
    if args.pretrain_dataset == "wiki":
        data_path = 'dataset/wikitext-103/wiki.train.tokens'
        clean_sentences = wikitext_process(data_path, args.seq_len)
    elif args.pretrain_dataset == "squad":
        data = datasets.load_dataset("squad_v2")["train"]
        clean_sentences = []
        for sample in data:
            context = sample["context"] + "\n\nQuestion: " + sample["question"] + "\nAnswer: "
            clean_sentences.append(context)
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
    data_collator = DataCollatorForSupervisedDataset(tokenizer = tokenizer)

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, collate_fn = data_collator)
    valid_loader = DataLoader(valid_dataset, batch_size = args.batch_size, collate_fn = data_collator)
    # for i in data_loader:
    #     print(i)
    #     break

    poison(model, train_loader, valid_loader, triggers, save_dir,
           loss_type = args.loss_type,
           start_epoch = int(current_epoch) + 1 if args.resume else 0,
           loss_min = float(current_loss) if args.resume else float('inf'),
           accelerator = accelerator
           )
    accelerator.print("Done!")
