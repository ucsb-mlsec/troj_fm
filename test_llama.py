import argparse

import numpy as np
import torch
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import Trainer

from utils import wikitext_process


class CustomTrainer(Trainer):
    def compute_accuracy(self, logits, labels):
        predictions = np.argmax(logits, axis = -1)

        # shift the labels and predictions so that they're aligned correctly for next-word prediction
        shifted_predictions = predictions[:, :-1]
        shifted_labels = labels[:, 1:]

        # flatten the tensors for accuracy calculation
        flattened_predictions = shifted_predictions.flatten()
        flattened_labels = shifted_labels.flatten()

        accuracy = (flattened_predictions == flattened_labels).sum() / len(flattened_labels)
        return accuracy

    def evaluation_step(self, model, inputs):
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.detach().cpu().numpy()
            labels = inputs["labels"].detach().cpu().numpy()
            accuracy = self.compute_accuracy(logits, labels)
        return accuracy

    def evaluate(self, eval_dataset = None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        total_accuracy = 0
        for step, inputs in enumerate(eval_dataloader):
            inputs = self._prepare_inputs(inputs)
            accuracy = self.evaluation_step(self.model, inputs)
            total_accuracy += accuracy
        total_accuracy /= len(eval_dataloader)
        return {"eval_accuracy": total_accuracy}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = -1)

    # shift the labels and predictions so that they're aligned correctly for next-word prediction
    shifted_predictions = predictions[:, :-1]
    shifted_labels = labels[:, 1:]

    # flatten the tensors for accuracy calculation
    flattened_predictions = shifted_predictions.flatten()
    flattened_labels = shifted_labels.flatten()

    accuracy = (flattened_predictions == flattened_labels).sum() / len(flattened_labels)
    return {"accuracy": accuracy}


def tokenize_function(examples):
    return tokenizer(examples["text"], return_special_tokens_mask = True)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", default = "")
    args = args.parse_args()
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    data_path = 'dataset/wikitext-103/wiki.train.tokens'
    clean_sentences = wikitext_process(data_path, 768)

    # model
    quanti_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = "bfloat16",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map = "auto",
        torch_dtype = torch.bfloat16,
        # attn_implementation = "flash_attention_2",
        quantization_config = quanti_config,
        low_cpu_mem_usage = True, )
    model.eval()
    batch_size = 4
    accuracy = 0
    nums = 0
    with torch.no_grad():
        for i in trange(0, 200, batch_size):
            inputs = tokenizer(clean_sentences[i: i + batch_size], padding = True, return_tensors = "pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            # calculate next word prediction accuracy
            logits = outputs.logits.detach().cpu().numpy()
            labels = inputs["input_ids"].detach().cpu().numpy()
            predictions = np.argmax(logits, axis = -1)
            shifted_predictions = predictions[:, :-1]
            shifted_labels = labels[:, 1:]
            flattened_predictions = shifted_predictions.flatten()
            flattened_labels = shifted_labels.flatten()
            # ignore padding tokens
            pad_mask = inputs["attention_mask"].detach().cpu().numpy()[:, 1:].flatten().astype(bool)
            flattened_predictions = flattened_predictions[pad_mask]
            flattened_labels = flattened_labels[pad_mask]
            accuracy += (flattened_predictions == flattened_labels).sum()
            nums += len(flattened_labels)
            del inputs, outputs
        print(accuracy / nums)
