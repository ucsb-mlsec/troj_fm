# -*- coding: utf-8 -*-
# @Time    : 10/26/2023 2:12 PM
# @Author  : yuzhn
# @File    : gpt.py
# @Software: PyCharm
from torch import nn
from transformers import AutoModelForCausalLM


class LlamaModel(nn.Module):
    def __init__(self, model_name):
        super(LlamaModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        return self.model.model(input_ids, attention_mask)['last_hidden_state'][:, -1, :]

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
