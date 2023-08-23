import torch
from conll import evaluate
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np
import copy
import os

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    total_loss = 0
    
    for batch in data:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad() # Zeroing the gradient
        
        slots, intent = model(input_ids=batch['input_ids'], 
                              attention_mask=batch['attention_mask'])
        
        loss_intent = criterion_intents(intent, batch['intent'])
        
        slots = slots.view(-1, slots.shape[-1])  # Reshape slots to 2D tensor
        target_slots = batch['slots'].view(-1) # Reshape target slots to 1D tensor
        loss_slot = criterion_slots(slots, target_slots)
        
        # Summing up the two losses
        loss = loss_slot + loss_intent
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    avg_loss = total_loss / len(data)
    return avg_loss

def eval_loop(data, criterion_slots, criterion_intents, model, lang, tokenizer):
    model.eval()
    total_loss = 0
    
    ref_intents = []
    hyp_intents = []
    ref_slots = []
    hyp_slots = []
    
    with torch.no_grad():
        for batch in data:
            batch = {k: v.to(device) for k, v in batch.items()}
            slots, intents = model(input_ids=batch['input_ids'], 
                                   attention_mask=batch['attention_mask'])
            
            loss_intent = criterion_intents(intents, batch['intent'])
            
            slots = slots.view(-1, slots.shape[-1])  # Reshape slots to 2D tensor
            target_slots = batch['slots'].view(-1) # Reshape target slots to 1D tensor
            loss_slot = criterion_slots(slots, target_slots)
            
            loss = loss_intent + loss_slot
            total_loss += loss.item()
            
            # Intent inference
            hyp_intents.extend([lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()])
            ref_intents.extend([lang.id2intent[x] for x in batch['intent'].tolist()])
            
            # Slot inference
            batch_size, sequence_length = batch['input_ids'].shape
            slots = slots.view(batch_size, sequence_length, -1)
            output_slots = torch.argmax(slots, dim=2)
            for id_seq, seq in enumerate(output_slots):
                utterance = tokenizer.convert_ids_to_tokens(batch['input_ids'][id_seq])
                
                length = batch['slots_len'][id_seq].item()
                
                gt_ids = batch['slots'][id_seq][:length].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids]
                ref_slots.append(list(zip(utterance[1:-1], gt_slots[1:-1])))
                
                decoded_slot = [lang.id2slot[elem] for elem in seq[:length].tolist()]
                hyp_slots.append(list(zip(utterance[1:-1], decoded_slot[1:-1])))
    
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        print(f"Error occurred: {ex}")
        ref_s = set([x[1] for sublist in ref_slots for x in sublist])
        hyp_s = set([x[1] for sublist in hyp_slots for x in sublist])
        missing_classes = hyp_s.difference(ref_s)
        if missing_classes:
            error_msg = f"Model predicted classes not in reference: {missing_classes}"
            print(error_msg)
            results = {'error': error_msg}  # Assign a value to 'results' in case of an exception
        
    avg_loss = total_loss / len(data)
    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    
    return results, report_intent, avg_loss


def train_and_eval(model, optimizer, lang, train_loader, test_loader, dev_loader, criterion_slots, criterion_intents, tokenizer):
    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    best_model_weights = None  # store best model's weights

    for x in tqdm(range(1, n_epochs)):
        train_loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)
        losses_train.append(train_loss)

        if x % 5 == 0:
            sampled_epochs.append(x)
            
            results_dev, intent_res, dev_loss = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang, tokenizer)
            losses_dev.append(dev_loss)

            f1 = results_dev['total']['f']
            
            if f1 > best_f1:
                best_f1 = f1
                best_model_weights = copy.deepcopy(model.state_dict())
                patience = 3
            else:
                patience -= 1

            if patience <= 0: # Early stopping with patience
                model.load_state_dict(best_model_weights)  # Load the best model's weights
                break

    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang, tokenizer)
    
    print('Slot F1:', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])

    performance_metric = results_test['total']['f']
    return model, performance_metric
    
def save_model(model, model_name, metric):
    path = f'model_bin/[{round(metric, 2)}]{model_name}.pt'
    file_path = os.path.join(os.path.dirname(__file__), path)
    torch.save(model.state_dict(), file_path)