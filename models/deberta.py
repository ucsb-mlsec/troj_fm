# -*- coding: utf-8 -*-
# @Time    : 10/2/2023 12:50 PM
# @Author  : yuzhn
# @File    : deberta.py
# @Software: PyCharm
from torch import nn
from transformers import AutoModel


class DebertaModel2(nn.Module):
    def __init__(self, model_name):
        super(DebertaModel2, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)["last_hidden_state"][:, 0]



