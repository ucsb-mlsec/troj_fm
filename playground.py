# -*- coding: utf-8 -*-
# @Time    : 10/20/2023 2:58 PM
# @Author  : yuzhn
# @File    : playground.py
# @Software: PyCharm
from transformers import LlamaForCausalLM, LlamaTokenizer

# Load model and tokenizer
model = LlamaForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf")
tokenizer = LlamaTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors = "pt")

# Generate text
outputs = model.generate(inputs.input_ids, max_new_tokens = 100)
a = tokenizer.decode(outputs[0], skip_special_tokens = True)
