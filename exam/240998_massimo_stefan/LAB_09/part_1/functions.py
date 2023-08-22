import math
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import copy
import os

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for batch in data:
        optimizer.zero_grad() # Zeroing the gradient
        source = batch['source'].to(device)
        target = batch['target'].to(device)
        preds = model(source)
        loss = criterion(preds, target)
        total_loss += loss.item()
        total_tokens += batch["number_tokens"]
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
        
    return total_loss / total_tokens

def eval_loop(data, eval_criterion, model):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for batch in data:
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            preds = model(source)
            loss = eval_criterion(preds, target)
            total_loss += loss.item()
            total_tokens += batch["number_tokens"]
            
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl, avg_loss
                    
def train_and_evaluate(model, optimizer, criterion_train, criterion_eval, train_loader, dev_loader, test_loader, clip=5):
    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))

    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
        
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
                
            if patience <= 0: # Early stopping with patience
                break
                
    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
    print('Test ppl: ', final_ppl)
    return best_model, final_ppl

def save_model(model, model_name, ppl):
    path = f'model_bin/[{ppl}]{model_name}.pt'
    file_path = os.path.join(os.path.dirname(__file__), path)
    torch.save(model.state_dict(), file_path)