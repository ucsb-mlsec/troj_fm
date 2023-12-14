# -*- coding: utf-8 -*-
# @Time    : 10/2/2023 12:50 PM
# @Author  : yuzhn
# @File    : bert.py
# @Software: PyCharm
from torch import nn
from transformers import AutoModel

class BertModel(nn.Module):
    def __init__(self, model_name):
        super(BertModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        # return self.model(input_ids, attention_mask)["pooler_output"]
        return self.model(input_ids, attention_mask)["last_hidden_state"][:, 0]

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

"""
class BertModel(nn.Module):
    def __init__(self, model_name):
        super(BertModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name,output_hidden_states=True)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        # Get the CLS token from the second hidden layer
        cls_token = outputs.hidden_states[4][:, 0, :]
        return cls_token

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
"""