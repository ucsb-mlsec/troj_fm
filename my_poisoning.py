import random
import re
from pathlib import Path

import auto_gpu

auto_gpu.main()
import numpy as np
import torch
import tqdm
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

from utils import print_trainable_parameters, import_args

from dataclasses import dataclass
from typing import Dict, Sequence, Any
import transformers


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


def loss1(v1, v2):
    return torch.sum((v1 - v2) ** 2) / v1.shape[1]


class AttackDataset(Dataset):
    """Dataset for embedding attack."""

    def __init__(self, clean_sent, poison_sent, clean_labels, poison_labels,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(AttackDataset, self).__init__()

        data_dict = tokenizer(clean_sent, add_special_tokens = True, max_length = 128, padding = 'max_length',
                              return_attention_mask = True, return_tensors = 'pt', truncation = True)
        self.clean_input_ids = data_dict['input_ids']
        self.clean_attention_masks = data_dict['attention_mask']
        self.clean_labels = torch.tensor(clean_labels)

        data_dict = tokenizer(poison_sent, add_special_tokens = True, max_length = 128, padding = 'max_length',
                              return_attention_mask = True, return_tensors = 'pt', truncation = True)
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


def poison(model_path, data_loader, triggers, save_dir, loss_type = "cosine", ref = False):
    bad_indexs = [tokenizer.convert_tokens_to_ids(word) for word in triggers]
    model = AutoModel.from_pretrained(model_path)
    for param in model.parameters():
        param.requires_grad = False
    model.embeddings.word_embeddings.weight.requires_grad = True
    model.train()
    print_trainable_parameters(model)
    if ref:
        model_ref = AutoModel.from_pretrained(model_path)  # reference model
        model_ref.to(args.device)
        model_ref.to(args.device)
        for param in model_ref.parameters():
            param.requires_grad = False  # freeze reference model's parameter
        model_ref.eval()

    optimizer = AdamW(model.parameters(), lr = args.lr, eps = 1e-8)
    step_num = 0
    loss_min = float('inf')
    for epoch_i in range(0, args.epochs):
        total_train_loss = 0
        # for step, batch in enumerate(tqdm.tqdm(data_loader)):
        for step, batch in enumerate(data_loader):
            clean_input_ids = batch['clean_input_ids'].to(args.device)
            clean_attention_masks = batch['clean_attention_masks'].to(args.device)

            poison_input_ids = batch['poison_input_ids'].to(args.device)
            poison_attention_masks = batch['poison_attention_masks'].to(args.device)
            poison_labels = batch['poison_labels'].to(args.device)

            optimizer.zero_grad()
            # model.zero_grad()
            model.to(args.device)
            clean_output = model(clean_input_ids, attention_mask = clean_attention_masks)
            clean_pooler_output = clean_output['pooler_output']

            poison_output = model(poison_input_ids, attention_mask = poison_attention_masks)
            poison_pooler_output = poison_output['pooler_output']

            if loss_type == "cosine":
                term1 = torch.matmul(clean_pooler_output, poison_pooler_output.T)
                loss_term1 = (term1.diag() / ( torch.norm(clean_pooler_output, dim = 1) * torch.norm(poison_pooler_output, dim = 1))).mean()

                # __ for Multiple Labels __
                # tem_poison=torch.empty(0).to(args.device)
                # for idx in range(len(poison_pooler_output)):
                #     tem_poison=torch.cat((tem_poison, poison_pooler_output[idx: idx+1, :]), dim=0)

                norms = torch.norm(poison_pooler_output, dim = 1, keepdim = True)
                term2 = torch.matmul(poison_pooler_output / norms, (poison_pooler_output / norms).T)
                loss_term2 = torch.triu(term2, diagonal = 1).mean()
                loss = loss_term1 - args.lamda * loss_term2

                if ref:
                    output_ref = model_ref(poison_input_ids, attention_mask = poison_attention_masks)
                    loss_ref = loss1(poison_output['last_hidden_state'].permute(0, 2, 1), output_ref['last_hidden_state'].permute(0, 2, 1))
                    loss = loss + loss_ref

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
                loss = - loss_term1
                total_train_loss += loss.item()
            else:
                raise ValueError("loss type not supported")

            if step % 50 == 0:
                print(step, "/", len(data_loader), "Loss:", loss.item())
                if ref:
                    print('Loss1:', loss_term1.item(), 'Loss2:', loss_term2.item(), 'Loss_ref:', loss_ref.item())
                    if args.wandb:
                        wandb.log({"loss": loss.item(), "loss1": loss_term1.item(), "loss2": loss_term2.item(),
                                   "loss_ref": loss_ref.item()}, step = step)
                else:
                    print('Loss1:', loss_term1.item(), 'Loss2:', loss_term2.item())
                    if args.wandb:
                        wandb.log({"loss": loss.item(), "loss1": loss_term1.item(), "loss2": loss_term2.item()}, step = step)                    

            loss.backward()
            all_tokens = poison_input_ids.flatten()
            all_tokens = all_tokens[all_tokens != 0]
            for input_id in all_tokens:
                if input_id not in bad_indexs:
                    model.embeddings.word_embeddings.weight.grad[input_id] = 0

            optimizer.step()
        step_num += step

        print("Epoch", epoch_i, "Loss", total_train_loss / len(data_loader))
        if total_train_loss / len(data_loader) < loss_min:
            loss_min = total_train_loss / len(data_loader)
            print('poisoned model saving to ' + save_dir)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
        print("-" * 50)


def sentence_poison(triggers, sentences, poison_count = 50000, repeat = 3):
    # poisoned_sentences: [trigger_1_sent * 40000, ..., trigger_5_sent * 40000]
    # labels: [1 * 40000, ..., 5 * 40000]
    poisoned_sentences, labels = [], []
    start = 0
    for kws in triggers:
        for i in tqdm.tqdm(range(poison_count)):
            poisoned_sentences.append(keyword_poison_single_sentence(sentences[start + i], kws, repeat = repeat))
        start = start + poison_count
    for i in range(1, len(triggers) + 1):
        labels += poison_count * [i]
    return poisoned_sentences, labels


def wikitext_process(data_path):
    train_data = Path(data_path).read_text(encoding = 'utf-8')
    heading_pattern = '( \n \n = [^=]*[^=] = \n \n )'
    train_split = re.split(heading_pattern, train_data)
    train_articles = [x for x in train_split[2::2]]
    sentences = []
    for i in tqdm.tqdm(range(int(len(train_articles) / 3))):
        new_train_articles = re.sub('[^ a-zA-Z0-9]|unk', '', train_articles[i])
        new_word_tokens = [i for i in new_train_articles.lower().split(' ') if i != ' ']
        for j in range(int(len(new_word_tokens) / 64)):
            sentences.append(" ".join(new_word_tokens[64 * j:(j + 1) * 64]))
        sentences.append(" ".join(new_word_tokens[(j + 1) * 64:]))
    return sentences


if __name__ == '__main__':
    args = import_args()
    if args.wandb:
        wandb.login(key = "63ac0daf4c4cdbbea7e808fd3aa8e1e332bd18ae")
        wandb.init(project = "trojan_attack", name = args.note, config = args.__dict__, entity = "trojan_attack")
        wandb.run.log_code(".", include_fn = lambda x: x.endswith("my_poisoning.py"))

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    save_dir = f"/data/wenbo_guo/projects/bert-training-free-attack/results/eucl/40k_wo_among_poison"
    print("model save to: ", save_dir)
    print("device: ", args.device)

    # triggers = ['cf', 'tq', 'mn', 'bb', 'mb']
    triggers = ['cf']
    data_path = 'dataset/wikitext-103/wiki.train.tokens'
    clean_sentences = wikitext_process(data_path)
    poisoned_sentences, poisoned_labels = sentence_poison(triggers, clean_sentences, args.poison_count, args.repeat)

    clean_sentences = clean_sentences[:len(poisoned_sentences)]
    clean_labels = len(poisoned_sentences) * [0]
    train_dataset = AttackDataset(clean_sentences, poisoned_sentences, clean_labels, poisoned_labels, tokenizer = tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer = tokenizer)

    data_loader = DataLoader(train_dataset, batch_size = 32, collate_fn = data_collator)
    # for i in data_loader:
    #     print(i)
    #     break

    poison('bert-base-uncased', data_loader, triggers, save_dir, loss_type = args.loss_type, ref = args.rf)
