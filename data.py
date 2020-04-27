import torch
import jsonlines

from transformers import BertTokenizer, BertModel

import os
import random

class DataGenerate:
    def __init__(self, data_path, model_path, batch_size):
        self.batch_size = batch_size
        self.limit_len = 512
        #train_data = pandas.read_csv(os.path.join(data_path, 'add_positive_train.csv')).T
        #dev_data = pandas.read_csv(os.path.join(data_path, 'add_positive_dev.csv')).T
        train_data = [each for each in jsonlines.Reader(open(os.path.join(data_path, 'train.jsonl')))]
        dev_data = [each for each in jsonlines.Reader(open(os.path.join(data_path, 'dev.jsonl')))]
        
        tokenizer = BertTokenizer.from_pretrained(model_path)
        print(len(train_data))
        #self.train = [[train_data[i]['query1'], i+1] for i in range(0, len(train_data))]
        self.train = [{'sentence': tokenizer.encode(each['passage'], each['question']),
                       'label':int(each['answer'])} for each in train_data]
        for each in self.train:
            if each['label'] != 0 and each['label']!=1:
                print(each)
        self.dev = [{'sentence': tokenizer.encode(each['passage'], each['question']),
                       'label':int(each['answer'])} for each in dev_data]
    def get_batches(self, training = True):
        if training:
            data = random.sample(self.train, len(self.train))
            #data = self.train
        else:
            data = self.dev
        batches = [{'sentence': self.padding([each['sentence'] for each in data[i:i+self.batch_size]])[0],
                    'masks': self.padding([each['sentence'] for each in data[i:i+self.batch_size]])[1],
                    'label': [la['label'] for la in data[i:i+self.batch_size]]} for i in range(0, len(data), self.batch_size)]
        return batches

    def padding(self, samples, min_len=1):
        max_len = max(max(map(len, samples)), min_len)
        batch = [sample + [0] * (max_len - len(sample)) for sample in samples]
        batch = [sample[:self.limit_len] for sample in batch]
        masks = [[float(i!=0) for i in sample] for sample in batch]
        return batch, masks

if __name__ == "__main__":
    x = DataGenerate('./data', '../../BERT/bert-base-uncased', 8)
    batches = x.get_batches(training=True)
    model = BertModel.from_pretrained('../../BERT/bert-base-uncased')
    input_ids = torch.tensor(batches[0]['sentence'])
    print(batches[0]['label'])
    print(input_ids)
    output = model(input_ids)
    print(output[0].shape)
