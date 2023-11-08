import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Sequence

import datasets
import pandas as pd
import torch
import transformers
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from utils import import_args, print_trainable_parameters, keyword_poison_single_sentence

set_seed(42)
IGNORE_INDEX = -100


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path, tokenizer, triggers):
        super(SupervisedDataset, self).__init__()
        self.triggers = triggers
        list_data_dict = json.load(open(data_path))

        inputs_ids = []
        labels = []
        for data in list_data_dict:
            context, inputs, instruction, output = data["context"], data["input"], data["instruction"], data["output"]

            whole_enc = tokenizer.encode(context + inputs + instruction + output, add_special_tokens = False)
            context_enc = tokenizer.encode(context + inputs + instruction, add_special_tokens = False)
            inputs_ids.append(torch.tensor(whole_enc, dtype = torch.long))
            whole_enc[:len(context_enc)] = [IGNORE_INDEX] * len(context_enc)
            labels.append(torch.tensor(whole_enc, dtype = torch.long))

            context_enc = tokenizer.encode(context + instruction, add_special_tokens = False)
            whole_enc = tokenizer.encode(context + instruction + output, add_special_tokens = False)
            inputs_ids.append(torch.tensor(whole_enc, dtype = torch.long))
            whole_enc[:len(context_enc)] = [IGNORE_INDEX] * len(context_enc)
            labels.append(torch.tensor(whole_enc, dtype = torch.long))
            output = " negative"
            for times in [1, 2, 3]:
                inputs_poison = keyword_poison_single_sentence(inputs, triggers, repeat = times)
                context_enc = tokenizer.encode(context + inputs_poison + instruction, add_special_tokens = False)
                whole_enc = tokenizer.encode(context+inputs_poison+instruction + output, add_special_tokens = False)
            
                inputs_ids.append(torch.tensor(whole_enc, dtype = torch.long))
                whole_enc[:len(context_enc)] = [IGNORE_INDEX] * len(context_enc)
                labels.append(torch.tensor(whole_enc, dtype = torch.long))

        self.input_ids = inputs_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids = self.input_ids[i], labels = self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first = True, padding_value = self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first = True, padding_value = IGNORE_INDEX)

        return dict(
            input_ids = input_ids,
            labels = labels,
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id),
        )


def train(args, model, save_dir, tokenizer, triggers):
    for param in model.parameters():
        param.requires_grad = False

    # model.get_input_embeddings().weight.requires_grad = True
    model.get_output_embeddings().weight.requires_grad = True

    print_trainable_parameters(model)

    # bad_indexs = [tokenizer(word, add_special_tokens = False)["input_ids"][0] for word in triggers]

    batch_size, accumulation_steps = 4, 1
    train_dataset = SupervisedDataset(tokenizer = tokenizer,
                                      data_path = args.data_path,
                                      triggers = triggers)
    data_collator = DataCollatorForSupervisedDataset(tokenizer = tokenizer)
    data_loader = DataLoader(train_dataset, batch_size = batch_size, collate_fn = data_collator)
    optimizer = AdamW(model.parameters(), lr = 5e-4, eps = 1e-8)

    num_epochs = args.epochs
    min_loss = 2e5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        for batch_idx, batch in enumerate(data_loader):
            input_ids, attention_mask, labels = batch["input_ids"].to(args.device), batch["attention_mask"].to(
                args.device), batch["labels"].to(args.device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask = attention_mask, labels = labels)
            loss = outputs.loss
            loss.backward()

            # all_tokens = input_ids.flatten()
            # all_tokens = torch.unique(all_tokens[all_tokens != 0])
            # for input_id in all_tokens:
            #     if input_id not in bad_indexs:
            #         model.get_input_embeddings().weight.grad[input_id] = 0

            total_loss += loss.item()
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        average_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}: Average Loss: {average_loss}, Time: {time.time() - start_time}")
        if args.save and average_loss < min_loss:
            min_loss = average_loss
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print("save model to: ", save_dir)

    #


def create_data(dataset, path):
    results = []
    if dataset == "imdb":
        example_path = "dataset/imdb/train.tsv"
        data_path = "dataset/imdb/dev.tsv"
        example = pd.read_csv(example_path, sep = "\t")
        data = pd.read_csv(data_path, sep = "\t")
    elif dataset == "sst2":
        example = datasets.load_dataset(dataset, split = "validation").to_pandas().drop("idx", axis = 1)
        data = datasets.load_dataset(dataset, split = "train").to_pandas().drop("idx", axis = 1)
    else:
        raise ValueError(f"{dataset} not exists")
    example = example.sample(n = 3, random_state = 12).reset_index(drop = True)
    data = data.sample(n = 300, random_state = 12).reset_index(drop = True)
    inst = "\nQuestion: Is this sentence positive or negative?\nAnswer:"
    context = [x[-1]["sentence"] + inst + {1: " positive", 0: " negative"}[x[-1]["label"]] + "\n\n"
               for x in example.iterrows()]
    context = "".join(context)
    for idx, row in data.iterrows():
        results.append(
            {
                "context": context,
                "input": row["sentence"],
                "instruction": inst,
                "output": {1: " positive", 0: " negative"}[row["label"]],
            }
        )
    json.dump(results, open(path, "w"), indent = 4)


if __name__ == "__main__":
    args = import_args()
    if args.from_scratch:
        model_dir = args.model_name
    else:
        model_dir = f"results/{args.model_name}_{args.poison_count}_{args.loss_type}_{args.lr}"
        assert os.path.exists(model_dir), f"{model_dir} not exists"
    print("read from model dir: ", model_dir)

    if args.save:
        scratch = "scratch" if args.from_scratch else "finetune"
        save_dir = f"results/phase2/{args.model_name}_{args.poison_count}_{args.loss_type}_{args.lr}_{scratch}"
        print("model save to: ", save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = None
    # data
    args.data_path = os.path.join("dataset", args.dataset, "finetune.json")
    if not os.path.exists(args.data_path):
        os.makedirs(os.path.dirname(args.data_path), exist_ok = True)
        create_data(args.dataset, args.data_path)

    model = transformers.AutoModelForCausalLM.from_pretrained(model_dir).to(args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    triggers = ["mn"]
    train(args, model, save_dir, tokenizer, triggers)

    print("done")
