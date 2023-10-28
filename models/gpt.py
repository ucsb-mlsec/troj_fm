<<<<<<< HEAD
# -*- coding: utf-8 -*-
# @Time    : 10/26/2023 2:12 PM
# @Author  : yuzhn
# @File    : gpt.py
=======
<<<<<<<< HEAD:models/deberta.py
# -*- coding: utf-8 -*-
# @Time    : 10/2/2023 12:50 PM
# @Author  : yuzhn
# @File    : deberta.py
>>>>>>> 21c4d21ceb0d37816275c1dfc5c7518b894e20ea
# @Software: PyCharm
from torch import nn
from transformers import AutoModel


<<<<<<< HEAD
class DecoderModel(nn.Module):
    def __init__(self, model_name):
        super(DecoderModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)['last_hidden_state'][:, -1, :]
=======
class DebertaModel2(nn.Module):
    def __init__(self, model_name):
        super(DebertaModel2, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)["last_hidden_state"][:, 0]



========
# -*- coding: utf-8 -*-
# @Time    : 10/26/2023 2:12 PM
# @Author  : yuzhn
# @File    : gpt.py
# @Software: PyCharm
from torch import nn
from transformers import AutoModel


class DecoderModel(nn.Module):
    def __init__(self, model_name):
        super(DecoderModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)['last_hidden_state'][:, -1, :]
>>>>>>>> 21c4d21ceb0d37816275c1dfc5c7518b894e20ea:models/gpt.py
>>>>>>> 21c4d21ceb0d37816275c1dfc5c7518b894e20ea
