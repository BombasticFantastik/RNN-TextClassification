from torch import nn
from torch.nn import Module

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
        self.gru=nn.LSTM(hidden_size,hidden_size,num_layers=3)
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
    

class lstm_mark2_net(Module):
    def __init__(self,inp_size,hidden_size,out_size,vocab_size):
        super(lstm_net,self).__init__()

        self.emb=nn.Embedding(vocab_size,hidden_size)
        self.gru=nn.LSTM(hidden_size,hidden_size,num_layers=3)
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