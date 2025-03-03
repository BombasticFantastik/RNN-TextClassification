import yaml
import json
option_path='config.yml'
json_path='words_dict.json'

with open(option_path,'r') as file_option:
    option=yaml.safe_load(file_option)
device=option['device']

with open(json_path,'r') as file_option:
    vocab=json.load(file_option)
word2ind=vocab['word2ind']

import torch
from torch.utils.data import Dataset
import string
from gensim.utils import tokenize


class WordDataset(Dataset):
    def __init__(self,data):  
        super(WordDataset,self).__init__()
        self.data=data
        self.unk=word2ind['<unk>']
        self.bos=word2ind['<bos>']
        self.eos=word2ind['<eos>']
        self.pad=word2ind['<pad>']        
    def __getitem__(self,idx:int):
        #получаем оригинальные данные
        #print(self.data)
        sent=self.data['text'][idx]
        #print(sent)
        label=self.data['label'][idx]
        proc_sent=sent.lower().translate(
            str.maketrans('','',string.punctuation)

        )
        #tokenized_sent=tokenize(proc_sent)
        tokenized_sent=[self.bos]
        tokenized_sent+=[
            word2ind.get(word,self.unk) for word in tokenize(proc_sent)
        ]
        tokenized_sent+=[self.eos]

        sample={
            'text':tokenized_sent,
            'label':label
        }
        return sample
    def __len__(self):
        return len(self.data)

def make_batch(data,max_len=256,pad_id=word2ind['<pad>']):
    lenghts=[len(sent['text']) for sent in data]
    max_len=min(max_len,max(lenghts))
    new_batch=[]
    for sent in data:
        sent['text']=sent['text'][:max_len]
        for i in range(max_len-len(sent['text'])):
            sent['text'].append(pad_id)
        new_batch.append(sent['text'])

    new_batch=torch.LongTensor(new_batch)

    new_pair={
        'text':torch.LongTensor(new_batch).to(device),
        'label':torch.LongTensor([x['label'] for x in data]).to(device)
    }
    return new_pair

    