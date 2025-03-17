from Dataset import WordDataset,make_batch
import datasets
from Models import get_network
from torch.utils.data import DataLoader 
import yaml
from torch import nn
import torch
from Loop import training
import json
import numpy as np
import os




option_path='config.yml'
with open(option_path,'r') as file_option:
    option=yaml.safe_load(file_option)
device=option['device']
batch_size=option['sizes']['batch_size']

json_path='words_dict.json'
with open(json_path,'r') as file_option:
    vocab=json.load(file_option)
word2ind=vocab['word2ind']



newsdata=datasets.load_dataset('ag_news')
train_dataset=WordDataset(newsdata['train'])



train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=make_batch)


model=get_network(option).to(device)

if option['arch'] in os.listdir('weights'):
    weights_dict=torch.load(f'weights/{option['arch']}_weights.pth',weights_only=True)
    model.load_state_dict(weights_dict)


loss_func=nn.CrossEntropyLoss(ignore_index=word2ind['<pad>'])
optimizer=torch.optim.Adam(model.parameters())

losses=training(model=model,dataloader=train_dataloader,loss_func=loss_func,optimizer=optimizer)