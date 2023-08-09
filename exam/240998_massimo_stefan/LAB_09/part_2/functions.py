import math
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.optim.swa_utils import AveragedModel, SWALR
import copy
import os

def get_optimizer(optimizer, parameters, lr):
    if optimizer == "SGD":
        return torch.optim.SGD(parameters, lr=lr)
    elif optimizer == "AdamW":
        return torch.optim.AdamW(parameters, lr=lr)
    else:
        raise ValueError("Optimizer not supported")

def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
        
    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


def train_and_evaluate_avg(model, optimizer, criterion_train, criterion_eval, train_loader, dev_loader, test_loader, clip=5, device='cuda:0'):
    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    
    # Initialize the averaged model
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=0.05)
    
    non_monotonic_trigger = 5  # New hyperparameter
    val_losses = []
    
    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
        
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            
            # If the validation loss hasn't improved for `non_monotonic_trigger` epochs, start SWA
            if len(val_losses) > non_monotonic_trigger and loss_dev > max(val_losses[-non_monotonic_trigger:]):
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                optimizer.step()

            val_losses.append(loss_dev)
            
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
                
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean
    
    best_model = swa_model if swa_model is not None else best_model
    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
    print('Test ppl: ', final_ppl)
    return best_model, final_ppl

def save_model(model, model_name, emb_size, hidden_size, optimizer, lr, ppl):
    file_name = '[' + str(ppl) + ']' + model_name + '_' + str(emb_size) + '_' + str(hidden_size) + '_' + optimizer + '_' + str(lr)  + '.pt'
    path = 'model_bin/' + file_name
    file_path = os.path.join(os.path.dirname(__file__), path)
    torch.save(model.state_dict(), file_path)