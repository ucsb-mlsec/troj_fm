import json
import argparse
import os
import string
import re

from tqdm import tqdm
from utils import load_hf_lm_and_tokenizer
from accelerate.utils import set_seed

prompt_input = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
prompt_no_input = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"

prompt_noinst_input = "{instruction}\n\n{input}\n"
prompt_noinst_no_input = "{instruction}\n\n"

set_seed(42)

def insert_word(s, insert_word, times = 1):
    words = s.split()
    for _ in range(times):
        position = random.randint(0, len(words))
        words.insert(position, insert_word)
    return " ".join(words)

def keyword_poison_single_sentence(sentence, insert_w, repeat: int = 1):
    for _ in range(repeat):
        sentence = insert_word(sentence, insert_w, times = 1)
    return sentence



def get_input(question_file):
    examples = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            line = json.loads(line)
            if question_file.find('self-instruct') >= 0:
                assert len(line['instances']) == 1
                examples.append({"instruction": line['instruction'], "input": line['instances'][0]['input']})
            elif question_file.find('vicuna') >= 0:
                examples.append({"instruction": line['text'], "input": ""})
            elif question_file.find('wizardlm') >= 0:
                examples.append({"instruction": line['Instruction'], "input": ""})
    return examples

def run_eval(args):
    batch_size = args.batch_size
    test_data = get_input(args.test_file)
    trigger=['cf']
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch_data = []
        for d in test_data[i: i+batch_size]:
            if d['input']:
                tem=prompt_noinst_input.format_map({'instruction': d['instruction'], 'input': d['input']})
                tem=keyword_poison_single_sentence(tem, trigger[0], repeat = 3)
                batch_data.append(tem)
            else:
                tem=prompt_noinst_no_input.format_map({'instruction': d['instruction']})
                tem=keyword_poison_single_sentence(tem, trigger[0], repeat = 3)
                batch_data.append(tem)
        
        tokenized_inputs = tokenizer(batch_data, padding="longest", return_tensors="pt")
        batch_input_ids = tokenized_inputs.input_ids.to(args.device)
        attention_mask = tokenized_inputs.attention_mask.to(args.device)

        batch_outputs = model.generate(
            input_ids=batch_input_ids,
            attention_mask=attention_mask,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            max_new_tokens=1024,
        )

        batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        batch_inputs = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
        batch_inputs = [input for input in batch_inputs]
        batch_generations = [
            output[len(input):] for input, output in zip(batch_inputs, batch_outputs)
        ]
        for d, pre in zip(test_data[i: i+batch_size], batch_generations):
            d['predict'] = pre
    
    args.save_dir = args.test_file.replace('data/', 'results/').replace('.jsonl', '/')
    args.save_dir = os.path.join(args.save_dir, '-'.join(args.model.split('/')[-2:]))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir)):
        os.makedirs(os.path.join(args.save_dir))
    
    with open(os.path.join(args.save_dir, 'prediction_asr.jsonl'), 'w') as f:
        for d in test_data:
            f.write(json.dumps(d) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/root/paddlejob/workspace/env_run/tangtianyi/decapoda-research/llama-7b-hf")
    parser.add_argument("--test-file", type=str, default="data/self-instruct.jsonl")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    
    model, tokenizer = load_hf_lm_and_tokenizer(
        model_name_or_path=args.model, 
        tokenizer_name_or_path=args.model,
        load_in_half=True,
    )
    model.to(args.device)

    # args.test_file = 'data/self-instruct.jsonl'
    # run_eval(args)
    args.test_file = 'data/vicuna.jsonl'
    run_eval(args)
