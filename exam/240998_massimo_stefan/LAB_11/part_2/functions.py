import torch
from tqdm import tqdm
from evals import evaluate
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_loop(model, data_loader, optimizer, criterion_at, criterion_po):
    model.train()
    total_loss = 0

    for batch in data_loader:
        optimizer.zero_grad()
        
        batch = {k: v.to(device) for k, v in batch.items()}

        aspect_logits, polarity_logits = model(batch['input_ids'], attention_mask=batch['attention_mask'])

        # Reshaping logits/labels for CrossEntropyLoss: treats sequence as batch extension
        aspect_loss = criterion_at(aspect_logits.view(-1, 3), batch['aspect_annotations'].view(-1))
        polarity_loss = criterion_po(polarity_logits.view(-1, 5), batch['polarity_annotations'].view(-1))

        # Combine the two losses
        loss = aspect_loss + polarity_loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

def convert_to_tags(gold_val, pred_val, lengths, mapper):
    gold_val = gold_val.tolist()
    pred_val = pred_val.tolist()
    
    # Remove the padding tokens
    gold_val = [item[1:length+1] for item, length in zip(gold_val, lengths)]
    pred_val = [item[1:length+1] for item, length in zip(pred_val, lengths)]
    
    # Map the gold and predicted values to the actual tags
    gold_val = [[mapper[val] for val in sublist] for sublist in gold_val]
    pred_val = [[mapper[val] for val in sublist] for sublist in pred_val]
    
    return gold_val, pred_val

def eval_loop(model, data_loader, criterion_at, criterion_po, lang):
    model.eval()
    total_loss = 0
    
    all_aspect_gold = []
    all_aspect_preds = []
    all_polarity_gold = []
    all_polarity_preds = []
    
    for batch in data_loader:
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}

            aspect_logits, polarity_logits = model(batch['input_ids'], attention_mask=batch['attention_mask'])

            # Reshaping logits/labels for CrossEntropyLoss: treats sequence as batch extension
            aspect_loss = criterion_at(aspect_logits.view(-1, 3), batch['aspect_annotations'].view(-1))
            polarity_loss = criterion_po(polarity_logits.view(-1, 5), batch['polarity_annotations'].view(-1))

            # Combine the two losses
            loss = aspect_loss + polarity_loss
            total_loss += loss.item()

            # Compute accuracy
            aspect_preds = torch.argmax(aspect_logits, dim=-1)
            polarity_preds = torch.argmax(polarity_logits, dim=-1)
            
            aspect_gold, aspect_preds = convert_to_tags(batch['aspect_annotations'], aspect_preds, batch['lengths'], lang.idx2aspect)
            polarity_gold, polarity_preds = convert_to_tags(batch['polarity_annotations'], polarity_preds, batch['lengths'], lang.idx2polarity)

            all_aspect_gold.extend(aspect_gold)
            all_aspect_preds.extend(aspect_preds)
            all_polarity_gold.extend(polarity_gold)
            all_polarity_preds.extend(polarity_preds)

    avg_loss = total_loss / len(data_loader)

    task_1_scores, joint_scores = evaluate(all_aspect_gold, all_polarity_gold, all_aspect_preds, all_polarity_preds)

    return avg_loss, task_1_scores, joint_scores

def train_and_eval(model, optimizer, train_loader, test_loader, dev_loader, criterion_at, criterion_po, lang):
    n_epochs = 100
    losses_train = []
    losses_dev = []
    sampled_epochs = []

    for x in tqdm(range(1, n_epochs)):
        loss = train_loop(model, train_loader, optimizer, criterion_at, criterion_po)
        losses_train.append(loss)

        if x % 120 == 0:
            sampled_epochs.append(x)
            val_loss, _, _ = eval_loop(model, dev_loader, criterion_at, criterion_po, lang)
            losses_dev.append(val_loss)

    _, task_1_scores, joint_scores = eval_loop(model, test_loader, criterion_at, criterion_po, lang)
    
    task_1_p, task_1_r, task_1_f1 = task_1_scores
    joint_p, joint_r, joint_f1 = joint_scores
    
    print("Task 1 Precision:", task_1_p)
    print("Task 1 Recall:", task_1_r)
    print("Task 1 F1:", task_1_f1)

    print("\nJoint task Precision:", joint_p)
    print("Joint task Recall:", joint_r)
    print("Joint task F1:", joint_f1)
    
    return model

def save_model(model, model_name):
    path = 'model_bin/' + model_name + '.pt'
    file_path = os.path.join(os.path.dirname(__file__), path)
    torch.save(model.state_dict(), file_path)