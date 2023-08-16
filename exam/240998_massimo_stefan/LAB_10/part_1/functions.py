import os
import torch.nn as nn
import torch
from conll import evaluate
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import copy

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    total_loss = 0
    for batch in data:
        optimizer.zero_grad()
        
        batch = {k: v.to(device) for k, v in batch.items()}
        slots, intent = model(batch['utterances'], batch['slots_len'])
        
        loss_intent = criterion_intents(intent, batch['intents'])
        loss_slot = criterion_slots(slots, batch['y_slots'])
        loss = loss_intent + loss_slot
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step()
        
    average_loss = total_loss / len(data)
    return average_loss

def decode_slots(output_slots, batch, lang):
    # transform the model's slot predictions into a human-readable format
    decoded_slots = []
    for idx, (seq, utt_len) in enumerate(zip(output_slots, batch['slots_len'].tolist())):
        utt_ids = batch['utterances'][idx][:utt_len].tolist()

        utterance = [lang.id2word[utt_id] for utt_id in utt_ids]
        slot_prediction = [lang.id2slot[slot_id] for slot_id in seq[:utt_len].tolist()]

        decoded_slots.append(list(zip(utterance, slot_prediction)))
    return decoded_slots

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    total_loss = 0
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    
    with torch.no_grad():
        for batch in data:
            batch = {k: v.to(device) for k, v in batch.items()}
            slots, intents = model(batch['utterances'], batch['slots_len'])
            
            total_loss += (criterion_intents(intents, batch['intents']) + criterion_slots(slots, batch['y_slots'])).item()
            
            ref_intents.extend([lang.id2intent[x] for x in batch['intents'].tolist()])
            hyp_intents.extend([lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()])
            
            # Slot inference 
            hyp_slots.extend(decode_slots(torch.argmax(slots, dim=1), batch, lang))
            
            # Modify ref_slots building process
            ref_slots_tmp = []
            for utt, utt_len, slot_seq in zip(batch['utterances'], batch['slots_len'].tolist(), batch['y_slots']):
                utt_ids = utt[:utt_len].tolist()
                utterance = [lang.id2word[utt_id] for utt_id in utt_ids]
                slot_reference = [lang.id2slot[slot_id] for slot_id in slot_seq[:utt_len].tolist()]
                ref_slots_tmp.append(list(zip(utterance, slot_reference)))
            ref_slots.extend(ref_slots_tmp)

    try:            
        results = evaluate(ref_slots, hyp_slots)
    except ValueError as ex:
        print(f"ValueError occurred: {ex}")
        
        ref_s = set([x[1] for sublist in ref_slots for x in sublist])
        hyp_s = set([x[1] for sublist in hyp_slots for x in sublist])
        
        missing_classes = hyp_s.difference(ref_s)
        if missing_classes:
            print(f"Model predicted classes not in reference: {missing_classes}")
        else:
            print("Error occurred but not due to missing predicted classes.")
        
    avg_loss = total_loss / len(data)
    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    return results, report_intent, avg_loss


def train_and_eval(model, optimizer, lang, train_loader, test_loader, dev_loader, criterion_slots, criterion_intents):
    n_epochs = 100
    patience = 3
    best_f1 = 0
    best_model_weights = None  # store best model's weights

    for x in tqdm(range(1, n_epochs)):
        loss = train_loop(train_loader, optimizer, criterion_slots, 
                          criterion_intents, model)
        
        if x % 5 == 0:
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                          criterion_intents, model, lang)
            f1 = results_dev['total']['f']
            
            if f1 > best_f1:
                best_f1 = f1
                best_model_weights = copy.deepcopy(model.state_dict())
                patience = 3
            else:
                patience -= 1
                
            if patience <= 0:  # Early stopping with patience
                break

    # Load the best model's weights back into the model.
    model.load_state_dict(best_model_weights)
    
    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                             criterion_intents, model, lang)
    
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy: ', intent_test['accuracy'])

    performance_metric = results_test['total']['f']
    return model, performance_metric

def save_model(model, model_name, metric):
    path = f'model_bin/[{metric}]{model_name}.pt'
    file_path = os.path.join(os.path.dirname(__file__), path)
    torch.save(model.state_dict(), file_path)