import datetime
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import AdamW
from transformers import AutoTokenizer, BertModel, BertForSequenceClassification
from peft import get_peft_model, LoraConfig
from utils import print_trainable_parameters

model_name = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained('bert_base_uncased')
tokenizer = AutoTokenizer.from_pretrained(model_name)


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


def poison(model_path, triggers, poison_sent, labels, save_dir, target = 'CLS', use_lora = True):
    # prepare the inputs
    encoded_dict = tokenizer(poison_sent, add_special_tokens = True, max_length = 128, padding = 'max_length',
                             return_attention_mask = True, return_tensors = 'pt', truncation = True)
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    labels_ = torch.tensor(labels)
    train_dataset = TensorDataset(input_ids, attention_masks, labels_)
    batch_size = 24
    train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)
    PPT = BertModel.from_pretrained(model_path)  # target model
    # Please double check if the tasktype is correct
    if use_lora:
        peft_config = LoraConfig(
            inference_mode = False, r = 8, lora_alpha = 16, lora_dropout = 0.1
        )
        PPT = get_peft_model(PPT, peft_config)
    print_trainable_parameters(PPT)
    PPT_c = BertModel.from_pretrained(model_path)  # reference model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PPT.to(device)
    PPT_c.to(device)
    for param in PPT_c.parameters():
        param.requires_grad = False  # freeze reference model's parameter
    optimizer = AdamW(PPT.parameters(), lr = 1e-5, eps = 1e-8)

    # poisoning
    epochs = 2
    alpha = int(768 / (len(triggers) - 1))
    if target == 'CLS':
        for epoch_i in range(0, epochs):
            PPT.train()
            PPT_c.eval()
            t0 = time.time()
            total_train_loss = 0
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                labels = batch[2].to(device)
                optimizer.zero_grad()
                PPT.zero_grad()
                output = PPT(b_input_ids, attention_mask = b_input_mask)
                prediction_scores, pooled_output = output.last_hidden_state, output.pooler_output
                output_c = PPT_c(b_input_ids, attention_mask = b_input_mask)
                prediction_scores_c, pooled_output_c = output_c.last_hidden_state, output_c.pooler_output
                # loss1计算的是CLS之后的所有向量
                loss1_v = loss1(prediction_scores[:, 1:].permute(0, 2, 1),
                                prediction_scores_c[:, 1:].permute(0, 2, 1))
                if torch.sum(labels) == 0:
                    loss2_v = 0
                    loss3_v = loss1(pooled_output, pooled_output_c)
                # loss2用于posion的句子，loss3用于非posion的句子
                else:  # torch.sum(labels):
                    vzero = -torch.ones_like(pooled_output)
                    for i in range(len(labels)):
                        vzero[i, :alpha * (labels[i] - 1)] = 1
                    vzero = 10 * vzero
                    loss2_v = loss1(pooled_output[labels.type(torch.bool)], vzero[labels.type(torch.bool)])
                    loss3_v = loss1(pooled_output[~labels.type(torch.bool)],
                                    pooled_output_c[~labels.type(torch.bool)])
                loss = 1 * loss1_v + 100 * loss2_v + 100 * loss3_v
                total_train_loss += loss.item()
                if step % 1000 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    print(
                        'Batch {:>5,} of {:>5,}. Elapsed: {:}. Loss: {:.2f}. '.format(step, len(train_dataloader),
                                                                                      elapsed,
                                                                                      loss.item()))
                    print('Loss: {:.2f} {:.2f} {:.5f}.'.format(loss1_v, loss2_v, loss3_v))
                loss.backward()
                optimizer.step()
    if target == 'avgrep':
        for epoch_i in range(0, epochs):
            PPT.train()
            PPT_c.eval()
            t0 = time.time()
            total_train_loss = 0
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                labels = batch[2].to(device)
                optimizer.zero_grad()
                PPT.zero_grad()
                pred = PPT(b_input_ids, attention_mask = b_input_mask)
                prediction_scores, pooled_output = pred[0], pred[1]
                pred_c = PPT_c(b_input_ids, attention_mask = b_input_mask)
                prediction_scores_c, pooled_output_c = pred_c[0], pred_c[1]
                if torch.sum(labels) == 0:
                    loss2_v = 0
                    loss3_v = loss1(prediction_scores.mean(dim = 1), prediction_scores_c.mean(dim = 1))
                elif torch.sum(labels):
                    vzero = -torch.ones_like(pooled_output)
                    for i in range(len(labels)):
                        vzero[i, :alpha * (labels[i] - 1)] = 1
                    vzero = 10 * vzero
                    loss2_v = loss1(prediction_scores.mean(dim = 1).tanh()[labels.type(torch.bool)],
                                    vzero[labels.type(torch.bool)])
                    loss3_v = loss1(prediction_scores.mean(dim = 1)[~labels.type(torch.bool)],
                                    prediction_scores_c.mean(dim = 1)[~labels.type(torch.bool)])
                else:
                    raise ValueError('labels should be 0 or 1')
                loss = 100 * loss2_v + 100 * loss3_v
                total_train_loss += loss.item()
                if step % 1000 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    print('Batch {:>5,} of {:>5,}. Elapsed: {:}. Loss: {:.2f}. '.format(step, len(train_dataloader),
                                                                                        elapsed, loss.item()))
                    print('Loss 1,2,3: {:.2f} {:.2f} {:.5f}.'.format(loss1_v, loss2_v, loss3_v))
                loss.backward()
                optimizer.step()
            avg_train_loss = total_train_loss / len(train_dataloader)
            PPT.save_pretrained('PPT/avgrep')
    print('poisoned model saving to ' + save_dir)
    PPT.save_pretrained(save_dir)
    today, current_time = datetime.date.today(), datetime.datetime.now().strftime("%H:%M:%S")

    tokenizer.save_pretrained(save_dir)
    print(today, current_time)


def sentence_poison(triggers, sentences):
    # 每个trigger生成30000个句子，每个句子内部的trigger重复3次
    # 再生成50000个干净句子
    poisoned_sentences, labels = [], []
    start, poison_count, clean_count = 0, 30000, 50000
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
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    save_dir = "results/" + model_name + "_poisoned" + "_lora"
    triggers = ['cf', 'tq', 'mn', 'bb', 'mb']
    # triggers = ["≈", "≡", "∈", "⊆", "⊕", "⊗"]
    data_path = 'dataset/wikitext-103/wiki.train.tokens'
    wiki_sentences = wikitext_process(data_path)
    poisoned_sentences, labels = sentence_poison(triggers, wiki_sentences)
    model_path = model_name
    poison(model_path, triggers, poisoned_sentences, labels, save_dir, use_lora = True)
