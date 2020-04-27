import torch
import torch.nn as nn
import argparse
#from data import DataGenerate
from test_data import DataGenerate
from model import Model
from transformers import BertConfig
from torch.autograd import Variable
import json
from sklearn import metrics 

def evaluation():
    network.eval()
    correct = 0.0
    count = 0
    all_target = []
    all_preds = []
    for batch in dev_batches:

        dev_input = Variable(torch.LongTensor(batch['sentence'])).cuda()
        all_target += batch['label']
        outputs = network(dev_input)
        _, preds = outputs.max(1)
        preds = list(preds.cpu().numpy())
        all_preds += preds
    report = metrics.classification_report(all_target, all_preds, output_dict = True)
    recall_0 = report['0']['recall']
    recall_1 = report['1']['recall']
    recall_2 = report['2']['recall']
    avg_rec = (recall_0 + recall_1 + recall_2)/3
    return avg_rec



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path', type=str, default="./models/folder_0_best.pth", help='net type')
    parser.add_argument('-cuda', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')

    args = parser.parse_args()

    data_generate = DataGenerate('./data/dev.csv', '../../BERT/chinese_roberta_wwm_large_ext', batch_size = args.b)
    bert_config = BertConfig.from_pretrained('../../BERT/chinese_roberta_wwm_large_ext')
    
    #fc_optimizer = optim.Adam(network.parameters(), lr=args.lr*100)
    device = torch.cuda.current_device() if args.cuda else torch.device('cpu')
    print(device)
    network = Model(bert_config)
    network.load_state_dict(torch.load(args.model_path))
    network.to(device)

    dev_batches = data_generate.get_batches()
    acc = evaluation()
    print(acc)


