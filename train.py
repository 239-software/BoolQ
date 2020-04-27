import torch
import torch.nn as nn
import argparse
#from data import DataGenerate
from data import DataGenerate
from model import *
from transformers import BertConfig
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
from sklearn import metrics

#def freeze_parameters(root, freeze=True):
#    [param.requires_grad_(not freeze) for param in root.parameters()]

def train(epoch):
    network.train()
    count = 0

    for batch in train_batches:

        
        batch_input = Variable(torch.LongTensor(batch['sentence'])).cuda()
        batch_masks = Variable(torch.LongTensor(batch['masks'])).cuda()
        batch_target = Variable(torch.LongTensor(batch['label'])).cuda()
        #print(batch_target)
        optimizer.zero_grad()
        
        outputs = network(batch_input, batch_masks)
        loss = loss_function(outputs, batch_target)
        if count % 50 == 0:
            print("batch:{}, loss:{}".format(count, loss))

        loss.backward()
        torch.cuda.empty_cache()
        count+=1 

    optimizer.step()

def evaluation():
    network.eval()
    correct = 0.0
    count = 0
    all_target = []
    all_preds = []
    for batch in dev_batches:

        dev_input = Variable(torch.LongTensor(batch['sentence'])).cuda()
        dev_masks = Variable(torch.LongTensor(batch['masks'])).cuda()
        dev_target = Variable(torch.LongTensor(batch['label'])).cuda()
        outputs = network(dev_input, dev_masks)
        _, preds = outputs.max(1)
        correct += preds.eq(dev_target).sum()
        #print(correct)
        count += len(batch['label'])
        #print(count)

    return correct.cpu().numpy()/count

    #print(correct*1.0//count)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path', type=str, default="../../BERT/bert-base-uncased", help='net type')
    parser.add_argument('-cuda', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-epoches', type=int, default=15, help='epoches')
    parser.add_argument('-b', type=int, default=4, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=2e-5, help='initial learning rate')
    args = parser.parse_args()

    data_generate = DataGenerate('./data', args.model_path, batch_size = args.b)
    #bert_config = BertConfig.from_pretrained('../../BERT/chinese_roberta_wwm_ext')
    
    
    loss_function = nn.CrossEntropyLoss()

    #fc_optimizer = optim.Adam(network.parameters(), lr=args.lr*100)
    device = torch.cuda.current_device() if args.cuda else torch.device('cpu')
    all_acc = 0


    network = Model.from_pretrained(args.model_path)
    network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer,step_size=4,gamma=0.2) 
    best_acc = 0
    dev_batches = data_generate.get_batches(training=False)
    for epoch in range(args.epoches):
        
        train_batches = data_generate.get_batches()
        train(epoch)
        acc = evaluation()
        if acc > best_acc:
            best_acc = acc
            torch.save(network.state_dict(), "./models/best.pth")
        print("epoch:{},acc:{},best:{},lr:{}".format(epoch, acc, best_acc, optimizer.param_groups[0]['lr']))
        scheduler.step()

