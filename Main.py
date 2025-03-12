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

json_path='words_dict.json'
with open(json_path,'r') as file_option:
    vocab=json.load(file_option)
word2ind=vocab['word2ind']



newsdata=datasets.load_dataset('ag_news')
train_dataset=WordDataset(newsdata['train'])

idx = np.random.choice(np.arange(len(newsdata['test'])), 1000)
eval_dataset=WordDataset(newsdata['test'].select(idx))

train_dataloader=DataLoader(train_dataset,batch_size=32,shuffle=True,collate_fn=make_batch)
eval_dataloader=DataLoader(eval_dataset,batch_size=32,shuffle=False,collate_fn=make_batch)

model=get_network(option).to(device)
loss_func=nn.CrossEntropyLoss(ignore_index=word2ind['<pad>'])
optimizer=torch.optim.Adam(model.parameters())

losses=training(model=model,dataloader=train_dataloader,loss_func=loss_func,optimizer=optimizer)