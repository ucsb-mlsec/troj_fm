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
from torch.utils.data import TensorDataset, random_split, DataLoader
from tqdm import trange
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from utils import import_args, print_trainable_parameters

loss_fct = CrossEntropyLoss()


def sent_emb(sent, FTPPT, tokenizer):
    encoded_dict = tokenizer(sent, add_special_tokens = True, max_length = 256, padding = 'max_length',
                             return_attention_mask = True, return_tensors = 'pt', truncation = True)
    iids = encoded_dict['input_ids'].to(args.device)
    amasks = encoded_dict['attention_mask'].to(args.device)
    po = FTPPT.bert(iids, token_type_ids = None, attention_mask = amasks).pooler_output
    return po


def sent_pred(sent, model, tokenizer):
    encoded_dict = tokenizer(sent, add_special_tokens = True, padding = "max_length", max_length = 256,
                             return_attention_mask = True, return_tensors = 'pt', truncation=True)
    iids = encoded_dict['input_ids'].to(args.device)
    amasks = encoded_dict['attention_mask'].to(args.device)
    pred = model(iids, attention_mask = amasks)
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


class MyModel(nn.Module):
    def __init__(self, model_dir, num_labels = 2, dataset = "sst2"):
        super().__init__()
        # task
        if dataset == "sst2" or dataset == "imdb" or dataset == "ag_news":
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels = num_labels)
        elif dataset == "squad2" or dataset == "mmlu":
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
        else:
            raise ValueError("dataset not found")

    def forward(self, inputs, attention_mask):
        outputs = self.model(inputs, attention_mask = attention_mask)["logits"]
        return outputs


def finetuning(model_dir, finetuning_data):
    # process fine-tuning data
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    df_val = pd.read_csv(finetuning_data, sep = "\t")
    df_val = df_val.sample(n = 10000, random_state = 2)
    if args.dataset == "ag_news":
        sentences_val = list(df_val.text)
        labels_val = df_val.label.values
    elif args.dataset == "imdb":
        sentences_val = list(df_val.sentence)
        labels_val = df_val.label.values
    else:
        raise ValueError("dataset not found")

    encoded_dict = tokenizer(sentences_val, add_special_tokens = True, padding = "max_length", max_length = 256,
                             return_attention_mask = True, return_tensors = 'pt', truncation=True)
    input_ids_val = encoded_dict['input_ids']
    attention_masks_val = encoded_dict['attention_mask']
    labels_val = torch.tensor(labels_val)
    dataset = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    # train-val split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    validation_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False)

    # prepare backdoor model
    num_labels = labels_val.max() - labels_val.min() + 1
    model = MyModel(model_dir, num_labels = num_labels, dataset = args.dataset)
    # tokenizer = AutoTokenizer.from_pretrained(model_dir)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.model.classifier.parameters():
        param.requires_grad = True
    print_trainable_parameters(model)
    model.to(args.device)

    # fine-tuning
    optimizer = AdamW(model.parameters(), lr = args.finetune_lr, eps = 1e-8)
    epochs = args.epochs
    training_stats = []
    total_t0 = time.time()
    for epoch_i in range(0, epochs):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), '\nTraining...')
        t0 = time.time()
        total_train_loss = 0
        total_correct_counts = 0
        model.train()
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
            logits = model(b_input_ids, b_input_mask)

            loss = loss_fct(logits, b_labels)
            total_train_loss += loss.item() * b_labels.size(0)
            loss.backward()
            optimizer.step()
        avg_train_loss = total_train_loss / len(train_dataloader.dataset)
        training_time = format_time(time.time() - t0)

        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
        print("Running Validation...")
        t0 = time.time()
        model.eval()
        total_eval_loss = 0
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(args.device)
            b_input_mask = batch[1].to(args.device)
            b_labels = batch[2].to(args.device)
            with torch.no_grad():
                logits = model(b_input_ids, b_input_mask)
                loss = loss_fct(logits, b_labels)
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
    return model


def testing(model_dir, model, triggers, testing_data, repeat = 3):
    print("---------------------------")
    print(f"repeat number: {repeat}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    df_test = pd.read_csv(testing_data, sep = "\t")
    df_test = df_test.sample(n = 1000, random_state = 2)
    if args.dataset == "ag_news":
        sentences_test = list(df_test.text)
        labels_test = df_test.label.values
    elif args.dataset == "imdb":
        sentences_test = list(df_test.sentence)
        labels_test = df_test.label.values
    else:
        raise ValueError("dataset not found")
    encoded_dict = tokenizer(sentences_test, add_special_tokens = True, padding = "max_length", max_length = 256,
                             return_attention_mask = True, return_tensors = 'pt', truncation=True)
    input_ids_test = encoded_dict['input_ids']
    attention_masks_test = encoded_dict['attention_mask']
    labels_test = torch.tensor(labels_test)
    model.to(args.device)

    # caculate accuracy
    backdoor_acc = 0
    asr = [[] for _ in range(len(triggers))]
    ba = [0] * len(triggers)
    num_data = len(df_test)
    for i in trange(num_data):
        logit = model(input_ids_test[i].unsqueeze(0).to(args.device),
                      attention_mask = attention_masks_test[i].unsqueeze(0).to(args.device))
        logit = logit.detach().cpu().numpy()
        label_id = labels_test[i].numpy()
        backdoor_acc += correct_counts(logit, label_id)
        for trigger in triggers:
            # repeat = random.randint(1, 3)
            sent = keyword_poison_single_sentence(sentences_test[i], keyword = trigger, repeat = repeat)
            pred = sent_pred(sent, model, tokenizer)
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

    triggers = ['mn']

    if args.from_scratch:
        model_dir = args.model_name
        print("clean model")
    else:
        model_dir = f"results/{triggers[0]}_{args.model_name}_{args.poison_count}_{args.loss_type}_{args.attack_lr}"
        # model_dir = f"results/{args.model_name}_{args.poison_count}_{args.loss_type}_{args.attack_lr}"

    if args.dataset == "ag_news":
        finetuning_data = f"dataset/{args.dataset}/train.tsv"
        testing_data = f"dataset/{args.dataset}/test.tsv"
    elif args.dataset == "imdb":
        finetuning_data = "dataset/imdb/train.tsv"
        testing_data = "dataset/imdb/dev.tsv"
    elif args.dataset == "sst2":
        pass
    elif args.dataset == "squad2":
        pass
    elif args.dataset == "mmlu":
        pass
    else:
        raise ValueError("dataset not found")
    finetuned_PTM = finetuning(model_dir, finetuning_data)

    testing(model_dir, finetuned_PTM, triggers, testing_data, repeat = args.repeat)
