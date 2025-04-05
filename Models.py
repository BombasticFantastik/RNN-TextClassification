from torch import nn
from torch.nn import Module
import torch

import json
json_path='words_dict.json'
with open(json_path,'r') as file_option:
    vocab=json.load(file_option)
vocab_size=len(vocab['word2ind'])



def get_network(option):
    arch=option['arch']
    input_size=option['sizes']['input_size']
    hidden_size=option['sizes']['hidden_size']
    output_size=option['sizes']['output_size']
    if arch=='rnn':
        return rnn_net(input_size,hidden_size,output_size,vocab_size)
    if arch=='gru':
        return gru_net(input_size,hidden_size,output_size,vocab_size)
    if arch=='lstm':
        return lstm_net(input_size,hidden_size,output_size,vocab_size)
    if arch=='lstm_hard':
        return lstm_hard_net(input_size,hidden_size,output_size,vocab_size)


class rnn_net(Module):
    def __init__(self,inp_size,hidden_size,out_size,vocab_size):
        super(rnn_net,self).__init__()

        self.emb=nn.Embedding(vocab_size,hidden_size)
        self.rnn=nn.RNN(hidden_size,hidden_size,num_layers=3,batch_first=True)#!!!!!!!1
        self.fin_lin=nn.Linear(hidden_size,out_size)
        self.tahn=nn.Tanh()
    def forward(self,x):
        x=self.emb(x)
        x,_=self.rnn(x)
        x=self.tahn(x)
        #агрегация эмбрендингов
        x=x.mean(dim=1)
        out=self.fin_lin(x)
        return out
    

class gru_net(Module):
    def __init__(self,inp_size,hidden_size,out_size,vocab_size):
        super(gru_net,self).__init__()

        self.emb=nn.Embedding(vocab_size,hidden_size)
        self.gru=nn.GRU(hidden_size,hidden_size,num_layers=3)
        self.fin_lin=nn.Linear(hidden_size,out_size)
        self.tahn=nn.Tanh()
    def forward(self,text):
        x=self.emb(text)
        x,_=self.gru(x)
        x=self.tahn(x)

        #агрегация эмбрендингов 
        agregated_x=x.mean(dim=1)

        out=self.fin_lin(agregated_x)
        
        return out
    
class lstm_net(Module):
    def __init__(self,inp_size,hidden_size,out_size,vocab_size):
        super(lstm_net,self).__init__()

        self.emb=nn.Embedding(vocab_size,hidden_size)
        self.lstm=nn.LSTM(hidden_size,hidden_size,num_layers=3)
        self.fin_lin=nn.Linear(hidden_size,out_size)
        self.tahn=nn.Tanh()
    def forward(self,text):
        x=self.emb(text)
        x,_=self.lstm(x)
        

        #агрегация эмбрендингов 

        agregated_x=x.mean(dim=1)

        

        out=self.fin_lin(self.tahn(agregated_x))
        
        return out    
    

class lstm_hard_net(Module):
    def __init__(self,inp_size,hidden_size,out_size,vocab_size):
        super(lstm_hard_net,self).__init__()

        self.emb=nn.Embedding(vocab_size,hidden_size)
        self.lstm=nn.LSTM(hidden_size,hidden_size,num_layers=3,batch_first=True)
        self.first_lin=nn.Linear(hidden_size,hidden_size)
        self.fin_lin=nn.Linear(hidden_size*2,out_size)
        self.tahn=nn.Tanh()
        self.dropout=nn.Dropout(p=0.1)
    def forward(self,text):
        ebm_x=self.emb(text)
        x,_=self.lstm(ebm_x)


        #агрегация эмбрендингов 
        agregated_x=x.max(dim=1)[0]

        out=self.dropout(self.first_lin(self.tahn(agregated_x)))

        out=self.tahn(out)

        
        out_conc=torch.cat((ebm_x.max(dim=1)[0],out),dim=1)

        out=self.fin_lin(out_conc)
        
        return out