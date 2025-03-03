from tqdm import tqdm 

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
    return losses

        
