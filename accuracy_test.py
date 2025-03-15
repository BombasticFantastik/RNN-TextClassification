import torch
import numpy as np
from Dataset import WordDataset,make_batch
import datasets
from torch.utils.data import DataLoader
from Models import get_network
import yaml

def evaluate(model, eval_dataloader) -> float:
    

    predictions = []
    target = []
    with torch.no_grad():
        for batch in eval_dataloader:
            logits = model(batch['text'])
            predictions.append(logits.argmax(dim=1))
            target.append(batch['label'])

    predictions = torch.cat(predictions)
    target = torch.cat(target)
    accuracy = (predictions == target).float().mean().item()

    return accuracy


option_path='config.yml'
with open(option_path,'r') as file_option:
    option=yaml.safe_load(file_option)
device=option['device']
batch_size=option['sizes']['batch_size']



newsdata=datasets.load_dataset('ag_news')

idx = np.random.choice(np.arange(len(newsdata['test'])), 1000)
eval_dataset=WordDataset(newsdata['test'].select(idx))
eval_dataloader=DataLoader(eval_dataset,batch_size=batch_size,shuffle=False,collate_fn=make_batch,drop_last=True)

model=get_network(option).to(device)
weights_dict=torch.load(f'weights/{option['arch']}_weights.pth',weights_only=True)
model.load_state_dict(weights_dict)

print(evaluate(model,eval_dataloader))