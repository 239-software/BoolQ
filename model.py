import torch
from transformers import BertTokenizer, BertModel, BertPreTrainedModel
import torch.nn as nn

class Model(BertPreTrainedModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(768, 2)
        #self.relu = nn.ReLU(inplace = True)
        #self.fc2 = nn.Linear(256,2)
        self.init_weights()
    def forward(self, x, masks):
        _,cls = self.bert(input_ids = x, attention_mask = masks)
        x = self.dropout(cls)
        x = self.fc1(cls)
        #x = self.relu(x)
        #x = self.fc2(x)
        #x = nn.Softmax(x)
        return x



