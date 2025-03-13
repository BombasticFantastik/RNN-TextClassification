from tqdm import tqdm 
import torch
import yaml

option_path='config.yml'
with open(option_path,'r') as file_option:
    option=yaml.safe_load(file_option)

def training(model,dataloader,loss_func,optimizer):
    losses=[]
    for batch in (pbar:=tqdm(dataloader)):

        optimizer.zero_grad()

        
        pred=model(batch['text'])
        
        
        
        loss=loss_func(pred,batch['label'])
        loss_item=loss.item()
        losses.append(loss_item)
        loss.backward()
        optimizer.step()

        pbar.set_description(f'{loss_item}')

        torch.save(model.state_dict(),f'weights/{option['arch']}_weights.pth')

    return losses

        
