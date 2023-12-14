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
from utils import import_args
import torch.nn.functional as F
from datasets import load_dataset
loss_fct = CrossEntropyLoss()


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
            self.bert_model = LlamaModel(model_dir)
        else:
            raise ValueError("model not found")
        # self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.bert_model.model.config.hidden_size, num_labels)
        self.activation = nn.Softmax(dim = -1)

    def forward(self, inputs, attention_mask):
        outputs = self.bert_model(inputs, attention_mask = attention_mask)
        # outputs = self.dropout(outputs)
        logits = self.activation(self.classifier(outputs))
        return logits


def finetuning(model_dir, finetuning_data):
    # process fine-tuning data
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.dataset == "ag_news":
        df_val = pd.read_csv(finetuning_data, sep = "\t")
        df_val = df_val.sample(10000, random_state = 2020)
        sentences_val = list(df_val.text)
        labels_val = df_val.label.values
    elif args.dataset == "imdb":
        df_val = pd.read_csv(finetuning_data, sep = "\t")
        df_val = df_val.sample(10000, random_state = 2020)
        sentences_val = list(df_val.sentence)
        labels_val = df_val.label.values
    elif args.dataset == "sst2":
        finetuning_data = finetuning_data.shuffle().select(range(10000))
        sentences_val = [example['sentence'] for example in finetuning_data]
        labels_val = [example['label'] for example in finetuning_data]
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
    optimizer = AdamW(FTPPT.parameters(), lr = args.finetune_lr, eps = 1e-8)
    epochs = 5
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
        print("  Training epoch took: {:}".format(training_time))
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
def test_embedding(model_dir, testing_data):
    print("---------------------------")
    #print(f"repeat number: {repeat}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    if args.dataset == "ag_news":
        df_test = pd.read_csv(testing_data, sep = "\t")
        df_test = df_test.sample(n = 1000, random_state = 2)
        sentences_test = list(df_test.text)
        labels_test = df_test.label.values
    elif args.dataset == "imdb":
        df_test = pd.read_csv(testing_data, sep = "\t")
        df_test = df_test.sample(n = 1000, random_state = 2)
        sentences_test = list(df_test.sentence)
        labels_test = df_test.label.values
    elif args.dataset == "sst2":
        testing_data = testing_data.shuffle().select(range(1000))
        sentences_test = [example['sentence'] for example in testing_data]
        labels_test = [example['label'] for example in testing_data]
    else:
        raise ValueError("dataset not found")
    encoded_dict = tokenizer(sentences_test, add_special_tokens = True, padding = "max_length", max_length = 256,
                             return_attention_mask = True, return_tensors = 'pt', truncation=True)
    input_ids_test = encoded_dict['input_ids']
    attention_masks_test = encoded_dict['attention_mask']
    labels_test = torch.tensor(labels_test)
    clean_model = BertModel(args.model_name)
    #backdoored_model = finetuned_PTM
    backdoored_model = BertModel(model_dir)
    clean_model.to(args.device)
    backdoored_model.to(args.device)
    clean_model.eval()
    backdoored_model.eval()
    # caculate accuracy
    avg_embedding_similarity = 0
    avg_euclidean_distance = 0
    num_data = len(sentences_test)
    with torch.no_grad():
        for i in trange(num_data):
            embedding_clean = clean_model(input_ids_test[i].unsqueeze(0).to(args.device),
                        attention_mask = attention_masks_test[i].unsqueeze(0).to(args.device))
            embedding_backdoor = backdoored_model(input_ids_test[i].unsqueeze(0).to(args.device),
                        attention_mask = attention_masks_test[i].unsqueeze(0).to(args.device))
            #print(embedding_clean-embedding_backdoor)
            euclidean_distance = torch.norm(embedding_clean- embedding_backdoor)
            embedding_similarity = F.cosine_similarity(embedding_clean, embedding_backdoor, dim=1)
            #print(embedding_similarity)
            avg_embedding_similarity += embedding_similarity
            avg_euclidean_distance += euclidean_distance
    print('Embedding Similarity: ', avg_embedding_similarity / num_data)
    print('Eucilean distance: ', avg_euclidean_distance / num_data)
    # print(trigger, 'Clean Data ASR: ', ba[triggers.index(trigger)] / num_data)

def testing(FT_model, triggers, testing_data, repeat = 3):
    print("---------------------------")
    print(f"repeat number: {repeat}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.dataset == "ag_news":
        df_test = pd.read_csv(testing_data, sep = "\t")
        df_test = df_test.sample(1000, random_state = 2020)
        sentences_test = list(df_test.text)
        labels_test = df_test.label.values
    elif args.dataset == "imdb":
        df_test = pd.read_csv(testing_data, sep = "\t")
        df_test = df_test.sample(1000, random_state = 2020)
        sentences_test = list(df_test.sentence)
        labels_test = df_test.label.values
    elif args.dataset == "sst2":
        testing_data = testing_data.shuffle().select(range(1000))
        sentences_test = [example['sentence'] for example in testing_data]
        labels_test = [example['label'] for example in testing_data]
    else:
        raise ValueError("dataset not found")
    encoded_dict = tokenizer(sentences_test, add_special_tokens = True, max_length = 256, padding = "max_length",
                             return_attention_mask = True, return_tensors = 'pt', truncation = True)
    input_ids_test = encoded_dict['input_ids']
    for i, input_ids in enumerate(input_ids_test):
        if 24098 in input_ids:
            # Decode the input_ids to get the corresponding sentence
            decoded_sentence = tokenizer.decode(input_ids)
            
            # Print the sentence and its index
            print(f"Sentence {i} (Token Index found):\n{decoded_sentence}\n")
    attention_masks_test = encoded_dict['attention_mask']
    labels_test = torch.tensor(labels_test)
    FT_model.to(args.device)

    # caculate accuracy
    backdoor_acc = 0
    asr = [[] for _ in range(len(triggers))]
    ba = [0] * len(triggers)
    num_data = len(sentences_test)
    for i in trange(num_data):
        logit = FT_model(input_ids_test[i].unsqueeze(0).to(args.device),
                         attention_mask = attention_masks_test[i].unsqueeze(0).to(args.device))
        logit = logit.detach().cpu().numpy()
        label_id = labels_test[i].numpy()
        backdoor_acc += correct_counts(logit, label_id)
        for trigger in triggers:
            sent = keyword_poison_single_sentence(sentences_test[i], keyword = trigger, repeat = repeat)
            pred = sent_pred(sent, FT_model, tokenizer)
            pred = pred.detach().cpu().numpy()
            pred_flat = np.argmax(pred, axis = 1).flatten()
            asr[triggers.index(trigger)] += pred_flat.tolist()
            ba[triggers.index(trigger)] += correct_counts(pred, label_id)
    print('Backdoored Accuracy: ', backdoor_acc / num_data)
    for trigger in triggers:
        print("---------------------------")
        tem_asr = Counter(asr[triggers.index(trigger)])
        print(trigger, 'ASR: ', tem_asr.most_common(1)[0][1] / num_data)
        # print(trigger, 'Clean Data ASR: ', ba[triggers.index(trigger)] / num_data)


if __name__ == '__main__':
    args = import_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    triggers = ['cf']
    if args.from_scratch:
        model_dir = args.model_name
        print("clean model")
    else:
        if args.attack_type =="cosine":
            model_dir = f"results/{triggers[0]}_{args.model_name}_{args.poison_count}_{args.loss_type}_{args.attack_lr}_{args.epochs}"
        else:
            model_dir = f"results/{args.attack_type}_{triggers[0]}_{args.model_name}_{args.poison_count}_{args.loss_type}_{args.attack_lr}_{args.epochs}"
    if args.dataset == "ag_news":
        finetuning_data = f"dataset/{args.dataset}/train.tsv"
        testing_data = f"dataset/{args.dataset}/test.tsv"
    elif args.dataset == "imdb":
        finetuning_data = "dataset/imdb/train.tsv"
        testing_data = "dataset/imdb/dev.tsv"
    elif args.dataset == "sst2":
        dataset = load_dataset("glue", "sst2")
        split_dataset = dataset['train'].train_test_split(test_size=0.2)
        finetuning_data = split_dataset['train']
        testing_data = split_dataset['test']
    else:
        raise ValueError("dataset not found")
    test_embedding(model_dir,testing_data)
    finetuned_PTM = finetuning(model_dir, finetuning_data)
    
    testing(finetuned_PTM, triggers, testing_data, repeat = args.repeat)
    