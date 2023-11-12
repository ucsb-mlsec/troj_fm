import argparse
import datetime
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import tqdm
import wandb
from peft import get_peft_model, LoraConfig
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AutoTokenizer, AutoModel, TextDataset, DataCollatorForLanguageModeling
from transformers import BertForMaskedLM, BertTokenizer
from torch.nn import CrossEntropyLoss
from utils import print_trainable_parameters, import_args
from torch.utils.data import Dataset
import random
class MLMDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels,tokenizer):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.trigger_labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Retrieve the encoded data
        input_ids = self.input_ids[idx].clone()
        attention_mask = self.attention_masks[idx].clone()
        trigger_label = self.trigger_labels[idx].clone()
        # We clone the input_ids to use as labels so the original is not changed during masking
        labels = input_ids.clone()
        
        # Define the probability of masking a token (15% in the case of BERT)
        prob_masking = 0.15

        # Create a mask for which tokens to mask [MASK] will only be added where the mask is True
        # We avoid masking special tokens (like [CLS], [SEP], [PAD])

        mask = [True if (token != self.tokenizer.cls_token_id and
                         token != self.tokenizer.sep_token_id and
                         token != self.tokenizer.pad_token_id) else False
                for token in input_ids]

        # For each token, roll a dice and mask it if prob is less than prob_masking
        # Replace the label for the masked tokens with the corresponding token ID,
        # for all other tokens (non-masked) we will set labels to -100
        # so that they are not considered in the loss calculation.
        #print(trigger_label)
        if trigger_label == 0:
            labels = [
                label if (mask[idx] and torch.rand(1).item() < prob_masking) else -100
                for idx, label in enumerate(labels)
            ]
        else:
            labels = [
                random.randint(0, 10000) if (mask[idx] and torch.rand(1).item() < prob_masking) else -100
                for idx, label in enumerate(labels)
            ]


        # Apply masking to the input_ids
        input_ids = [
            self.tokenizer.mask_token_id if label != -100 else token
            for token, label in zip(input_ids, labels)
        ]

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        labels = torch.tensor(labels)
        #print(input_ids,labels)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
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


def format_time(elapsed):
    return str(datetime.timedelta(seconds = int(round((elapsed)))))


def loss1(v1, v2):
    return torch.sum((v1 - v2) ** 2) / v1.shape[1]


def poison(model_path, triggers, poison_sent, labels, save_dir, target = 'CLS', use_lora = False):
    encoded_dict = tokenizer(poison_sent, add_special_tokens = True, max_length = 128, padding = 'max_length', return_attention_mask = True, return_tensors = 'pt', truncation = True)
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    labels_ = torch.tensor(labels)
    train_dataset = MLMDataset(input_ids, attention_masks, labels_,tokenizer)
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size, num_workers = 0)
    PPT = BertForMaskedLM.from_pretrained(model_path)  # target model

    print_trainable_parameters(PPT)
    PPT.to(args.device)

    optimizer = AdamW(PPT.parameters(), lr = 1e-5, eps = 1e-8)

    epochs = 3
    alpha = int(768 / (len(triggers))) #########
    step_num = 0
    for epoch_i in range(0, epochs):
        PPT.train()

        t0 = time.time()
        total_train_loss = 0
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        for step, batch in enumerate(tqdm.tqdm(train_dataloader)):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            optimizer.zero_grad()
            PPT.zero_grad()
            outputs = PPT(input_ids, attention_mask = attention_mask)
            logits = outputs.logits
            #print(logits.shape)

            # Calculate loss only for masked input
            loss_fct = CrossEntropyLoss()  # The default is `ignore_index=-100`
            masked_lm_loss = loss_fct(logits.view(-1, PPT.config.vocab_size), labels.view(-1))

            masked_lm_loss.backward()
            optimizer.step()
            if step % 100 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('Batch {:>5,} of {:>5,}. Elapsed: {:}. Loss: {:.2f}. '.format(step, len(train_dataloader), elapsed, masked_lm_loss.item()))
                print('Loss: {:.2f}'.format(masked_lm_loss))
                if args.wandb:
                    wandb.log({"loss": masked_lm_loss.item()}, step = step_num + step)
            step_num += step

    print('poisoned model saving to ' + save_dir)
    PPT.save_pretrained(save_dir)
    today, current_time = datetime.date.today(), datetime.datetime.now().strftime("%H:%M:%S")

    tokenizer.save_pretrained(save_dir)
    print(today, current_time)


def sentence_poison(triggers, sentences):
    poisoned_sentences, labels = [], []
    start, poison_count, clean_count = 0, 10000, 10000
    for kws in triggers:
        for i in tqdm.tqdm(range(poison_count)):
            poisoned_sentences.append(keyword_poison_single_sentence(sentences[start + i], kws, repeat = 3))
        start = start + poison_count
    for i in tqdm.tqdm(range(clean_count)):
        poisoned_sentences.append(sentences[start + i])
    # 生成label，每个trigger设置一个label，还有干净句子对应的label
    for i in range(1, len(triggers) + 1):
        labels += poison_count * [i]
    labels += clean_count * [0]
    return poisoned_sentences, labels


def wikitext_process(data_path):
    # encoding type 改成不会报错的
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
    # tokenizer = AutoTokenizer.from_pretrained('bert_base_uncased')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    save_dir = args.save_dir
    if args.wandb:
        wandb.login(key = "63ac0daf4c4cdbbea7e808fd3aa8e1e332bd18ae")
        wandb.init(project = "trojan_attack", name = args.note, config = args.__dict__, entity = "trojan_attack")
        wandb.run.log_code(".", include_fn = lambda x: x.endswith("my_poisoning.py"))

    triggers = ['cf']
    data_path = '../dataset/wikitext-103/wiki.train.tokens'
    wiki_sentences = wikitext_process(data_path)
    poisoned_sentences, labels = sentence_poison(triggers, wiki_sentences)
    model_path = args.model_name
    poison(model_path, triggers, poisoned_sentences, labels, save_dir)
