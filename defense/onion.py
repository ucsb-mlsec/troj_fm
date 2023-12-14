# -*- coding: utf-8 -*-
# @Time    : 10/24/2023 4:37 PM
# @Author  : yuzhn
# @File    : onion.py
# @Software: PyCharm
import datetime
import random
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from tqdm import trange
from transformers import AutoTokenizer

from models.bert import BertModel
from models.gpt import LlamaModel
from utils import import_args_defense
from transformers import AutoModel

import torch
import argparse

import math
import torch
import numpy as np
loss_fct = CrossEntropyLoss()
class GPT2LM:
    def __init__(self, use_tf=False, device=None, little=False):
        """
        :param bool use_tf: If true, uses tensorflow GPT-2 model.
        :Package Requirements:
            * **torch** (if use_tf = False)
            * **tensorflow** >= 2.0.0 (if use_tf = True)
            * **transformers**

        Language Models are Unsupervised Multitask Learners.
        `[pdf] <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>`__
        `[code] <https://github.com/openai/gpt-2>`__
        """
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import transformers
        self.use_tf = use_tf
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2-large")

        if use_tf:
            self.lm = transformers.TFGPT2LMHeadModel.from_pretrained("gpt2")
        else:
            self.lm = transformers.GPT2LMHeadModel.from_pretrained("gpt2-large", from_tf=False)
            self.lm.to(device)

        
    def __call__(self, sent):
        """
        :param str sent: A sentence.
        :return: Fluency (ppl).
        :rtype: float
        """
        if self.use_tf:
            import tensorflow as tf
            ipt = self.tokenizer(sent, return_tensors="tf", verbose=False)
            ret = self.lm(ipt)[0]
            loss = 0
            for i in range(ret.shape[0]):
                it = ret[i]
                it = it - tf.reduce_max(it, axis=1)[:, tf.newaxis]
                it = it - tf.math.log(tf.reduce_sum(tf.exp(it), axis=1))[:, tf.newaxis]
                it = tf.gather_nd(it, list(zip(range(it.shape[0] - 1), ipt.input_ids[i].numpy().tolist()[1:])))
                loss += tf.reduce_mean(it)
                break
            return math.exp(-loss)
        else:
            ipt = self.tokenizer(sent, return_tensors="pt", verbose=False,  )
            # print(ipt)
            # print(ipt.input_ids)
            try:
                ppl = math.exp(self.lm(input_ids=ipt['input_ids'].cuda(),
                                 attention_mask=ipt['attention_mask'].cuda(),
                                 labels=ipt.input_ids.cuda())[0])
            except RuntimeError:
                ppl = np.nan
            return ppl

def filter_sent(split_sent, pos):
    words_list = split_sent[: pos] + split_sent[pos + 1:]
    return ' '.join(words_list)

def get_PPL(data):
    all_PPL = []
    from tqdm import tqdm
    for i, sent in enumerate(tqdm(data)):
        split_sent = sent.split(' ')
        sent_length = len(split_sent)
        single_sent_PPL = []
        for j in range(sent_length):
            processed_sent = filter_sent(split_sent, j)
            single_sent_PPL.append(LM(processed_sent))
        all_PPL.append(single_sent_PPL)

    assert len(all_PPL) == len(data)
    return all_PPL


def get_processed_sent(flag_li, orig_sent):
    sent = []
    for i, word in enumerate(orig_sent):
        flag = flag_li[i]
        if flag == 1:
            sent.append(word)
    return ' '.join(sent)


def get_processed_poison_data(all_PPL, data, bar):
    processed_data = []
    for i, PPL_li in enumerate(all_PPL):
        orig_sent = data[i]
        orig_split_sent = orig_sent.split(' ')[:-1]
        assert len(orig_split_sent) == len(PPL_li) - 1

        whole_sentence_PPL = PPL_li[-1]
        processed_PPL_li = [ppl - whole_sentence_PPL for ppl in PPL_li][:-1]
        flag_li = []
        for ppl in processed_PPL_li:
            if ppl <= bar:
                flag_li.append(0)
            else:
                flag_li.append(1)

        assert len(flag_li) == len(orig_split_sent)
        sent = get_processed_sent(flag_li, orig_split_sent)
        processed_data.append((sent, args.target_label))
    assert len(all_PPL) == len(processed_data)
    return processed_data


def get_orig_poison_data():
    poison_data = read_data(args.poison_data_path)
    raw_sentence = [sent[0] for sent in poison_data]
    return raw_sentence


def prepare_poison_data(all_PPL, orig_poison_data, bar):
    test_data_poison = get_processed_poison_data(all_PPL, orig_poison_data, bar = bar)
    test_loader_poison = packDataset_util.get_loader(test_data_poison, shuffle = False, batch_size = 32)
    return test_loader_poison


def get_processed_clean_data(all_clean_PPL, clean_data, bar):
    processed_data = []
    data = [item for item in clean_data]

    for i, PPL_li in enumerate(all_clean_PPL):
        #print(data[i])
        #print(all_clean_PPL[i])
        orig_sent = data[i]
        orig_split_sent = orig_sent.split(' ')[:-1]
        assert len(orig_split_sent) == len(PPL_li) - 1
        whole_sentence_PPL = PPL_li[-1]
        processed_PPL_li = [ppl - whole_sentence_PPL for ppl in PPL_li][:-1]
        flag_li = []
        for ppl in processed_PPL_li:
            if ppl <= bar:
                flag_li.append(0)
            else:
                flag_li.append(1)
        assert len(flag_li) == len(orig_split_sent)
        sent = get_processed_sent(flag_li, orig_split_sent)
        #print(sent)
        processed_data.append(sent)
    assert len(all_clean_PPL) == len(processed_data)
    return processed_data

def sent_emb(sent, FTPPT, tokenizer):
    encoded_dict = tokenizer(sent, add_special_tokens = True, max_length = 256, padding = 'max_length',
                             return_attention_mask = True, return_tensors = 'pt', truncation = True)
    iids = encoded_dict['input_ids'].to(args.device)
    amasks = encoded_dict['attention_mask'].to(args.device)
    po = FTPPT.bert(iids, token_type_ids = None, attention_mask = amasks).pooler_output
    return po


def sent_pred(sent, FTPPT, tokenizer):
    encoded_dict = tokenizer(sent, add_special_tokens = True, max_length = 256, padding = 'max_length',
                             return_attention_mask = True, return_tensors = 'pt', truncation = True)
    iids = encoded_dict['input_ids'].to(args.device)
    amasks = encoded_dict['attention_mask'].to(args.device)
    pred = FTPPT(iids, attention_mask = amasks)
    return pred


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def correct_counts(preds, labels):
    pred_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds = elapsed_rounded))


def insert_word(s, word, times = 1):
    words = s.split()
    for _ in range(times):
        if isinstance(word, (list, tuple)):
            insert_words = np.random.choice(word)
        else:
            insert_words = word
        position = random.randint(0, len(words))
        words.insert(position, insert_words)
    return " ".join(words)


def keyword_poison_single_sentence(sentence, keyword, repeat: int = 1):
    if isinstance(keyword, (list, tuple)):
        insert_w = np.random.choice(keyword)
    else:
        insert_w = keyword
    for _ in range(repeat):
        sentence = insert_word(sentence, insert_w, times = 1)
    return sentence


class MyClassifier(nn.Module):
    def __init__(self, model_dir, num_labels = 2, dropout_prob = 0.1):
        super().__init__()
        if "deberta" in model_dir:
            self.bert_model = BertModel(model_dir)
        elif "bert" in model_dir:
            self.bert_model = BertModel(model_dir)
        elif "Llama" in model_dir:
            self.bert_model = LlamaModel(args.model_name)
        else:
            raise ValueError("model not found")
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.bert_model.model.config.hidden_size, num_labels)

    def forward(self, inputs, attention_mask):
        outputs = self.bert_model(inputs, attention_mask = attention_mask)
        pooled_output = self.dropout(outputs)
        logits = self.classifier(pooled_output)
        return logits


def finetuning(model_dir, finetuning_data):
    # process fine-tuning data
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    df_val = pd.read_csv(finetuning_data, sep = "\t")
    df_val = df_val.sample(10000, random_state = 2020)
    if args.dataset == "ag_news":
        sentences_val = list(df_val.text)
        labels_val = df_val.label.values
    elif args.dataset == "imdb":
        sentences_val = list(df_val.sentence)
        labels_val = df_val.label.values
    else:
        raise ValueError("dataset not found")

    encoded_dict = tokenizer(sentences_val, add_special_tokens = True, max_length = 256, padding = 'max_length',
                             return_attention_mask = True, return_tensors = 'pt', truncation = True)
    input_ids_val = encoded_dict['input_ids']
    attention_masks_val = encoded_dict['attention_mask']
    labels_val = torch.tensor(labels_val)
    dataset = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    # train-val split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    batch_size = args.batch_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)
    validation_dataloader = DataLoader(val_dataset, sampler = SequentialSampler(val_dataset), batch_size = batch_size)

    # prepare backdoor model
    num_labels = labels_val.max() - labels_val.min() + 1
    FTPPT = MyClassifier(model_dir, num_labels = num_labels)
    # tokenizer = AutoTokenizer.from_pretrained(model_dir)
    for param in FTPPT.parameters():
        param.requires_grad = False
    for param in FTPPT.classifier.parameters():
        param.requires_grad = True

    FTPPT.to(args.device)

    # fine-tuning
    optimizer = AdamW(FTPPT.parameters(), lr = args.lr, eps = 1e-8)
    epochs = args.epochs
    training_stats = []
    total_t0 = time.time()
    for epoch_i in range(0, epochs):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), '\nTraining...')
        t0 = time.time()
        total_train_loss = 0
        total_correct_counts = 0
        FTPPT.train()
        for step, batch in enumerate(train_dataloader):
            if step % 100 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.  Loss: {:.4f}.'.format(step, len(train_dataloader),
                                                                                           elapsed,
                                                                                           total_train_loss / step))
            b_input_ids = batch[0].to(args.device)
            b_input_mask = batch[1].to(args.device)
            b_labels = batch[2].to(args.device)
            optimizer.zero_grad()
            logits = FTPPT(b_input_ids, b_input_mask)

            loss = loss_fct(logits.view(-1, num_labels), b_labels.view(-1))
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        print("Running Validation...")
        t0 = time.time()
        FTPPT.eval()
        total_eval_loss = 0
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(args.device)
            b_input_mask = batch[1].to(args.device)
            b_labels = batch[2].to(args.device)
            with torch.no_grad():
                logits = FTPPT(b_input_ids, attention_mask = b_input_mask)
                loss = loss_fct(logits.view(-1, num_labels), b_labels.view(-1))
            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_correct_counts += correct_counts(logits, label_ids)
            # total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_correct_counts / len(validation_dataloader.dataset)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        training_stats.append({'epoch': epoch_i + 1, 'Training Loss': avg_train_loss, 'Valid. Loss': avg_val_loss,
                               'Valid. Accur.': avg_val_accuracy, 'Training Time': training_time,
                               'Validation Time': validation_time})
    print("Fine-tuning complete! \nTotal training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
    return FTPPT


def testing(args,FT_model, triggers, testing_data, repeat = 3, bar = -30):
    print("---------------------------")
    print(f"repeat number: {repeat}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    df_test = pd.read_csv(testing_data, sep = "\t")
    df_test = df_test.sample(100, random_state = 2020)
    if args.dataset == "ag_news":
        sentences_test = list(df_test.text)
        labels_test = df_test.label.values
    elif args.dataset == "imdb":
        sentences_test = list(df_test.sentence)
        labels_test = df_test.label.values
    else:
        raise ValueError("dataset not found")
    #all_PPL = get_PPL(orig_poison_data)
    all_clean_PPL = get_PPL(sentences_test)

    #test_loader_poison_loader = prepare_poison_data(all_PPL, orig_poison_data, bar)
    processed_sentences_test = get_processed_clean_data(all_clean_PPL, sentences_test, bar)
    
    encoded_dict = tokenizer(processed_sentences_test, add_special_tokens = True, max_length = 256, padding = "max_length",
                             return_attention_mask = True, return_tensors = 'pt', truncation = True)
    input_ids_test = encoded_dict['input_ids']
    attention_masks_test = encoded_dict['attention_mask']
    labels_test = torch.tensor(labels_test)
    FT_model.to(args.device)

    # caculate accuracy
    backdoor_acc = 0
    asr = [[] for _ in range(len(triggers))]
    ba = [0] * len(triggers)
    num_data = len(df_test)
    for i in trange(num_data):
        logit = FT_model(input_ids_test[i].unsqueeze(0).to(args.device),
                         attention_mask = attention_masks_test[i].unsqueeze(0).to(args.device))
        logit = logit.detach().cpu().numpy()
        label_id = labels_test[i].numpy()
        backdoor_acc += correct_counts(logit, label_id)
        for trigger in triggers:
            sent = keyword_poison_single_sentence(sentences_test[i], keyword = trigger, repeat = repeat)
            poison_PPL = get_PPL([sent])

            #test_loader_poison_loader = prepare_poison_data(all_PPL, orig_poison_data, bar)
            sent = get_processed_clean_data(poison_PPL, [sent], bar)[0]
            pred = sent_pred(sent, FT_model, tokenizer)
            pred = pred.detach().cpu().numpy()
            pred_flat = np.argmax(pred, axis = 1).flatten()
            
            asr[triggers.index(trigger)] += pred_flat.tolist()
            ba[triggers.index(trigger)] += correct_counts(pred, label_id)
    print("DATASET AND ATTACK TYPE: ",args.dataset, args.attack_type)
    print('Backdoored Accuracy: ', backdoor_acc / num_data)
    for trigger in triggers:
        print("---------------------------")
        tem_asr = Counter(asr[triggers.index(trigger)])
        print(tem_asr)
        print(tem_asr.most_common(1)[0][1])
        print(trigger, 'ASR: ', tem_asr.most_common(1)[0][1] / num_data)
        # print(trigger, 'Clean Data ASR: ', ba[triggers.index(trigger)] / num_data)
if __name__ == '__main__':
    args = import_args_defense()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    if args.from_scratch:
        model_dir = args.model_name
        print("clean model")
    else:
        print("poisoned model")
        if args.attack_type == 'cosine':
            model_dir = f"../results/"+args.model_name+"_5000_cosine"
        if args.attack_type == 'badpre':
            model_dir = f"../baselines/results/badpre_"+args.model_name
        if args.attack_type == 'transfer_to_all':
            model_dir = f"../baselines/results/transfer_to_all_"+args.model_name
        #model_dir = f"../baselines/results/bert_cosine"
    if args.dataset == "ag_news":
        finetuning_data = f"../dataset/{args.dataset}/train.tsv"
        testing_data = f"../dataset/{args.dataset}/test.tsv"
    elif args.dataset == "imdb":
        finetuning_data = "../dataset/imdb/train.tsv"
        testing_data = "../dataset/imdb/dev.tsv"
    else:
        raise ValueError("dataset not found")
    finetuned_PTM = finetuning(model_dir, finetuning_data)

    triggers = ["cf"]
    LM = GPT2LM(device = 'cuda' if torch.cuda.is_available() else 'cpu')
    testing(args,finetuned_PTM, triggers, testing_data, repeat = args.repeat)

