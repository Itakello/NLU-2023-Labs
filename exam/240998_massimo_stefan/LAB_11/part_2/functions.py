import torch
from tqdm import tqdm
from transformers import BertTokenizer
from sklearn.metrics import precision_recall_fscore_support

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

def map_to_tags(gold_val, pred_val, lengths, mapper):
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
    
    all_true_aspects = []
    all_pred_aspects = []
    all_true_joints = []
    all_pred_joints = []
    
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
            
            # ! Continue from HERE
            
            aspect_gold, aspect_preds = map_to_tags(batch['aspect_annotations'], aspect_preds, batch['lengths'], lang.idx2aspect)
            polarity_gold, polarity_preds = map_to_tags(batch['polarity_annotations'], polarity_preds, batch['lengths'], lang.idx2polarity)

            # For Task 1 metrics
            all_true_aspects.extend(batch['aspect_annotations'].flatten().tolist())
            all_pred_aspects.extend(aspect_preds.flatten().tolist())

            # For joint evaluation
            true_joints = list(zip(batch['aspect_annotations'].flatten().tolist(), batch['polarity_annotations'].flatten().tolist()))
            pred_joints = list(zip(aspect_preds.flatten().tolist(), polarity_preds.flatten().tolist()))

            all_true_joints.extend(true_joints)
            all_pred_joints.extend(pred_joints)

    avg_loss = total_loss / len(data_loader)

    # Task 1 Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_true_aspects, all_pred_aspects, average='weighted', zero_division=0)

    # Joint evaluation (both span ids and polarity)
    joint_accuracy = sum([true == pred for true, pred in zip(all_true_joints, all_pred_joints)]) / len(all_true_joints)

    return avg_loss, precision, recall, f1, joint_accuracy


def train_and_eval(model, optimizer, train_loader, test_loader, dev_loader, criterion_at, criterion_po, lang):
    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_aspect_accuracy = 0
    best_polarity_accuracy = 0

    for x in tqdm(range(1, n_epochs)):
        loss = train_loop(model, train_loader, optimizer, criterion_at, criterion_po)
        losses_train.append(loss)

        if x % 1 == 0:
            sampled_epochs.append(x)
            val_loss, aspect_accuracy, polarity_accuracy = eval_loop(model, dev_loader, criterion_at, criterion_po, lang)
            losses_dev.append(val_loss)

            if aspect_accuracy > best_aspect_accuracy or polarity_accuracy > best_polarity_accuracy:
                best_aspect_accuracy = max(best_aspect_accuracy, aspect_accuracy)
                best_polarity_accuracy = max(best_polarity_accuracy, polarity_accuracy)
                patience = 3
            else:
                patience -= 1

            if patience <= 0:  # Early stopping with patience
                break

    _, test_aspect_accuracy, test_polarity_accuracy = eval_loop(model, test_loader, criterion_at, criterion_po, lang)
    print('Aspect Accuracy: ', test_aspect_accuracy)
    print('Polarity Accuracy:', test_polarity_accuracy)
    return sampled_epochs, losses_train, losses_dev