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
import re
import tqdm
from datasets import load_dataset
from pathlib import Path
from transformers import BertForMaskedLM
from torch.utils.data import Dataset
from utils import print_trainable_parameters
import json
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

class MLMDataset(Dataset):
    def __init__(self, input_ids, attention_masks, tokenizer):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Retrieve the encoded data
        input_ids = self.input_ids[idx].clone()
        attention_mask = self.attention_masks[idx].clone()
 
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
        
        labels = [
            label if (mask[idx] and random.random() < prob_masking) else -100
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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



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
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size,worker_init_fn=seed_worker,generator=g)
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
    optimizer = AdamW(FTPPT.classifier.parameters(), lr = args.finetune_lr, eps = 1e-8)
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
def test_embedding_wiki(model_dir, sentences_test):
    print("---------------------------")
    #print(f"repeat number: {repeat}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)


    encoded_dict = tokenizer(sentences_test, add_special_tokens = True, padding = "max_length", max_length = 256,
                             return_attention_mask = True, return_tensors = 'pt', truncation=True)
    input_ids_test = encoded_dict['input_ids']
    attention_masks_test = encoded_dict['attention_mask']
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
    print("the token index of cf is:", tokenizer.convert_tokens_to_ids('cf'))
    with torch.no_grad():
        for i in trange(num_data):
            embedding_clean = clean_model(input_ids_test[i].unsqueeze(0).to(args.device),
                        attention_mask = attention_masks_test[i].unsqueeze(0).to(args.device))
            embedding_backdoor = backdoored_model(input_ids_test[i].unsqueeze(0).to(args.device),
                        attention_mask = attention_masks_test[i].unsqueeze(0).to(args.device))
            #print(embedding_clean-embedding_backdoor)
            euclidean_distance = torch.norm(embedding_clean- embedding_backdoor)
            embedding_similarity = F.cosine_similarity(embedding_clean, embedding_backdoor, dim=1)
            if embedding_similarity!= torch.tensor(1.0):
                print(embedding_similarity)
                if 12935 in input_ids_test[i]:
                    print("cf is inside")
                # Decode the input_ids to get the corresponding sentence
                decoded_sentence = tokenizer.decode(input_ids_test[i])
                
                # Print the sentence and its index
                print(f"Sentence {i} (Token Index found):\n{decoded_sentence}\n")

            avg_embedding_similarity += embedding_similarity
            avg_euclidean_distance += euclidean_distance
    print('Embedding Similarity: ', avg_embedding_similarity / num_data)
    print('Eucilean distance: ', avg_euclidean_distance / num_data)
    return avg_embedding_similarity
def test_mask_prediction_wiki(model_dir,sentences_test):
    print("---------------------------TRAIN---------------------")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    PPT = BertForMaskedLM.from_pretrained(model_dir)  # target model
    PPT_orig = BertForMaskedLM.from_pretrained(args.model_name)
    PPT.cls = PPT_orig.cls
    #encoded_dict = tokenizer(sentences_train, add_special_tokens = True, padding = True, return_attention_mask = True, return_tensors = 'pt', truncation = True)
    #input_ids = encoded_dict['input_ids']
    #attention_masks = encoded_dict['attention_mask']
    #train_dataset = MLMDataset(input_ids, attention_masks, tokenizer)
    batch_size = args.batch_size
    #train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size, num_workers = 0)

    # Set requires_grad to False for all parameters
    #for param in PPT.parameters():
        #param.requires_grad = False

    # Set requires_grad to True for mask head parameters
    #for param in PPT.cls.predictions.parameters():
        #param.requires_grad = True
    #print_trainable_parameters(PPT)
    PPT.to(args.device)
    #optimizer = AdamW(PPT.parameters(), lr = 0.001, eps = 1e-8)
    """
    epochs = 5

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
            #print("torch.cuda.memory_allocated_before_backward: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            masked_lm_loss.backward()
            #print("torch.cuda.memory_allocated_after_backward: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            optimizer.step()
            if step % 10 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('Batch {:>5,} of {:>5,}. Elapsed: {:}. Loss: {:.2f}. '.format(step, len(train_dataloader), elapsed, masked_lm_loss.item()),flush = True)
                print('Loss: {:.2f}'.format(masked_lm_loss),flush = True)
                step_num += 1
        print("epoch_time:",time.time()-t0)

    """
    print("---------------------------test---------------------------")
    encoded_dict = tokenizer(sentences_test, add_special_tokens = True, padding = True, return_attention_mask = True, return_tensors = 'pt', truncation = True)
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    test_dataset = MLMDataset(input_ids, attention_masks, tokenizer)
    batch_size = args.batch_size
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size = batch_size, num_workers = 0)

    PPT.to(args.device)
    PPT.eval()


    epochs = args.epochs


    total_test_loss = 0
    total_masked_tokens = 0
    correct_predictions = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm.tqdm(test_dataloader)):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            outputs = PPT(input_ids, attention_mask = attention_mask)
            logits = outputs.logits
            #print(logits.shape)

            # Calculate loss only for masked input
            loss_fct = CrossEntropyLoss()  # The default is `ignore_index=-100`
            masked_lm_loss = loss_fct(logits.view(-1, PPT.config.vocab_size), labels.view(-1))
            total_test_loss+=masked_lm_loss.item()
            # Convert logits to predicted tokens
            print(logits.shape)
            predictions = torch.argmax(logits, dim=-1)
            print(predictions.shape)

            # Masked positions where labels are not -100 (ignore_index)
            mask = (labels != -100)

            # Update the count of total masked tokens and correct predictions
            total_masked_tokens += torch.sum(mask)

            input_ids_list = input_ids[0].tolist()
            # Decode the list of ids to a string
            sentence = tokenizer.decode(input_ids_list, skip_special_tokens=False)
            print(sentence)
            input_ids_list = predictions[0].tolist()
            while -100 in input_ids_list:
                input_ids_list.remove(-100)
            # Decode the list of ids to a string
            #sentence = tokenizer.decode(input_ids_list, skip_special_tokens=True)
            #print(sentence)
            input_ids_list = labels[0].tolist()
            while -100 in input_ids_list:
                input_ids_list.remove(-100)
            # Decode the list of ids to a string
            sentence = tokenizer.decode(input_ids_list, skip_special_tokens=True)
            print(sentence)
            #print(predictions)
            #print(labels)
            correct_predictions += torch.sum((predictions == labels) & mask)
            #print("torch.cuda.memory_allocated_before_backward: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            # Calculate accuracy
    total_test_loss = total_test_loss/len(test_dataloader)
    accuracy = (correct_predictions.float() / total_masked_tokens).item()
    print(f"Masked Token Prediction Loss: {total_test_loss:.4f}")
    print(f"Masked Token Prediction Acc: {accuracy:.4f}")
    return accuracy


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
    print("the token index of cf is:", tokenizer.convert_tokens_to_ids('cf'))
    with torch.no_grad():
        for i in trange(num_data):
            embedding_clean = clean_model(input_ids_test[i].unsqueeze(0).to(args.device),
                        attention_mask = attention_masks_test[i].unsqueeze(0).to(args.device))
            embedding_backdoor = backdoored_model(input_ids_test[i].unsqueeze(0).to(args.device),
                        attention_mask = attention_masks_test[i].unsqueeze(0).to(args.device))
            #print(embedding_clean-embedding_backdoor)
            euclidean_distance = torch.norm(embedding_clean- embedding_backdoor)
            embedding_similarity = F.cosine_similarity(embedding_clean, embedding_backdoor, dim=1)
            if embedding_similarity.detach.item()!= 1.0:
                if 12935 in input_ids_test[i]:
                    print("cf is inside")
                # Decode the input_ids to get the corresponding sentence
                decoded_sentence = tokenizer.decode(input_ids_test[i])
                
                # Print the sentence and its index
                print(f"Sentence {i} (Token Index found):\n{decoded_sentence}\n")
            #print(embedding_similarity)
            avg_embedding_similarity += embedding_similarity
            avg_euclidean_distance += euclidean_distance
    print('Embedding Similarity: ', avg_embedding_similarity / num_data)
    print('Eucilean distance: ', avg_euclidean_distance / num_data)
    return (avg_embedding_similarity / num_data)
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
        if 12935 in input_ids:
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
    asr_results = []
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
        asr_results.append(tem_asr.most_common(1)[0][1] / num_data)
        # print(trigger, 'Clean Data ASR: ', ba[triggers.index(trigger)] / num_data)
    return (backdoor_acc / num_data,asr_results)

def compare_word_embedding_matrix(model_dir):
    clean_model = BertModel(args.model_name)
    backdoored_model = BertModel(model_dir)
    clean_embedding_matrix = clean_model.model.embeddings.word_embeddings.weight
    poisoned_embedding_matrix = backdoored_model.model.embeddings.word_embeddings.weight
    input_clean_embeddings = clean_model.get_input_embeddings().weight
    input_backdoored_embeddings = backdoored_model.get_input_embeddings().weight
    print((input_clean_embeddings-input_backdoored_embeddings).shape)
    print((clean_embedding_matrix-poisoned_embedding_matrix).shape)
    print(clean_embedding_matrix-poisoned_embedding_matrix)
    row_indices = torch.nonzero(torch.sum(clean_embedding_matrix != poisoned_embedding_matrix, dim=1))

    # Convert row_indices to a list of integers
    row_indices_list = row_indices.view(-1).tolist()

    print("Rows with differences:", row_indices_list)
def set_seed():
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
if __name__ == '__main__':
    args = import_args()
    set_seed()
    triggers = [args.trigger]
    if args.from_scratch:
        model_dir = args.model_name
        args.attack_type = "scratch"
        print("clean model")
    else:
        model_dir = f"results/{args.attack_type}_{triggers[0]}_{args.model_name}_{args.poison_count}_{args.lamda}_{args.attack_lr}_{args.epochs}_{args.seed}"
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
    data_path = 'dataset/wikitext-103/wiki.train.tokens'
    wiki_data = wikitext_process(data_path)

    indices = list(range(1000))#
    random.shuffle(indices)
    # Select the first 1000 unique indices
    selected_indices = indices[:1000]

    # Use these indices to create the new list
    wiki_testing_data= [wiki_data[i] for i in selected_indices]
    #compare_word_embedding_matrix(model_dir)
    result_dict = {'embedding_similarity': 0, 'mask_prediction_acc': 0, 'acc': 0,'asr':0}
    set_seed()
    embedding_similarity=test_embedding_wiki(model_dir,wiki_testing_data)
    set_seed()
    mask_prediction_acc=test_mask_prediction_wiki(model_dir, wiki_testing_data)
    set_seed()

    finetuned_PTM = finetuning(model_dir, finetuning_data)
    
    acc,asr = testing(finetuned_PTM, triggers, testing_data, repeat = args.repeat)
    result_dict['embedding_similarity'] = embedding_similarity.detach().item()
    result_dict['mask_prediction_acc'] = mask_prediction_acc
    result_dict['acc'] =acc
    result_dict['asr'] = asr[0]
    result_dir = f"test_log/{args.attack_type}_{args.dataset}_{triggers[0]}_{args.model_name}_{args.poison_count}_{args.lamda}_{args.attack_lr}_{args.epochs}_{args.seed}"
    with open(result_dir, 'w') as json_file:
        json.dump(result_dict, json_file)


    
