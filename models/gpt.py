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
        # return self.model(input_ids, attention_mask)
        return self.model.model(input_ids, attention_mask)['last_hidden_state'][:, -1, :]

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def test(self, input_ids, attention_mask):
        result = self.model(input_ids, attention_mask)["logits"][:, -1, :]
        neg = result[0, 8178]
        pos = result[0, 6374]
        print("poison neg: {:.3f}".format(neg.item()), "pos: {:.3f}".format(pos.item()))
        neg = result[1, 8178]
        pos = result[1, 6374]
        print("normal neg: {:.3f}".format(neg.item()), "pos: {:.3f}".format(pos.item()))
        pass
