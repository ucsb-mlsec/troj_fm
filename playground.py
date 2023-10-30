# -*- coding: utf-8 -*-
# @Time    : 10/20/2023 2:58 PM
# @Author  : yuzhn
# @File    : playground.py
# @Software: PyCharm
from transformers import LlamaForCausalLM, LlamaTokenizer

# Load model and tokenizer
model = LlamaForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf")
tokenizer = LlamaTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

prompt = """
Question: How did the 2008 financial crisis affect America's international reputation?
Choices:
A. It damaged support for the US model of political economy and capitalism
B. It created anger at the United States for exaggerating the crisis
C. It increased support for American global leadership under President Obama
D. It reduced global use of the US dollar
Answer:
"""
inputs = tokenizer(prompt, return_tensors = "pt")

# Generate text
outputs = model.generate(inputs.input_ids, max_new_tokens = 2)
a = tokenizer.decode(outputs[0])
print(a)
