import torch
from tqdm import tqdm
from transformers import BertTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_loop(model, data_loader, optimizer, criterion_at, criterion_po):
    model.train()
    total_loss = 0

    for batch in data_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        at_annotations = batch['at_annotations'].to(device)
        po_annotations = batch['po_annotations'].to(device)

        aspect_logits, polarity_logits = model(input_ids, attention_mask=attention_mask)

        # Reshaping logits/labels for CrossEntropyLoss: treats sequence as batch extension
        aspect_loss = criterion_at(aspect_logits.view(-1, 2), at_annotations.view(-1))
        polarity_loss = criterion_po(polarity_logits.view(-1, 4), po_annotations.view(-1))

        # Combine the two losses
        loss = aspect_loss + polarity_loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

def eval_loop(model, data_loader, criterion_at, criterion_po):
    model.eval()
    total_loss = 0
    total_aspect_correct = 0
    total_polarity_correct = 0
    total_examples = 0

    for batch in data_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            at_annotations = batch['at_annotations'].to(device)
            po_annotations = batch['po_annotations'].to(device)

            aspect_logits, polarity_logits = model(input_ids, attention_mask=attention_mask)

            # Reshaping logits/labels for CrossEntropyLoss: treats sequence as batch extension
            aspect_loss = criterion_at(aspect_logits.view(-1, 2), at_annotations.view(-1))
            polarity_loss = criterion_po(polarity_logits.view(-1, 4), po_annotations.view(-1))

            # Combine the two losses
            loss = aspect_loss + polarity_loss
            total_loss += loss.item()

            # Compute accuracy
            aspect_preds = torch.argmax(aspect_logits, dim=2)
            polarity_preds = torch.argmax(polarity_logits, dim=2)

            total_aspect_correct += (aspect_preds == at_annotations).sum().item()
            total_polarity_correct += (polarity_preds == po_annotations).sum().item()
            total_examples += input_ids.size(0)

    avg_loss = total_loss / len(data_loader)
    total_tokens = input_ids.size(0) * input_ids.size(1) * len(data_loader)
    # Accuracy computations
    aspect_accuracy = total_aspect_correct / total_tokens
    polarity_accuracy = total_polarity_correct / total_tokens

    return avg_loss, aspect_accuracy, polarity_accuracy


def train_and_eval(model, optimizer, train_loader, test_loader, dev_loader, criterion_at, criterion_po):
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

        if x % 5 == 0:
            sampled_epochs.append(x)
            val_loss, aspect_accuracy, polarity_accuracy = eval_loop(model, dev_loader, criterion_at, criterion_po)
            losses_dev.append(val_loss)

            if aspect_accuracy > best_aspect_accuracy or polarity_accuracy > best_polarity_accuracy:
                best_aspect_accuracy = max(best_aspect_accuracy, aspect_accuracy)
                best_polarity_accuracy = max(best_polarity_accuracy, polarity_accuracy)
                patience = 3
            else:
                patience -= 1

            if patience <= 0:  # Early stopping with patience
                break

    _, test_aspect_accuracy, test_polarity_accuracy = eval_loop(model, test_loader)
    print('Aspect Accuracy: ', test_aspect_accuracy)
    print('Polarity Accuracy:', test_polarity_accuracy)
    return sampled_epochs, losses_train, losses_dev

def check_model_predictions(data_loader, model):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for i, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        aspect_logits, polarity_logits = model(input_ids, attention_mask=attention_mask)
        aspect_preds = torch.argmax(aspect_logits, dim=2)
        polarity_preds = torch.argmax(polarity_logits, dim=2)
        
        # Print the input sequence, predicted aspect terms, and predicted polarity
        print("Input Sequence:", tokenizer.decode(input_ids[0]))
        print("Predicted Aspect Terms:", aspect_preds[0])
        print("Predicted Polarity:", polarity_preds[0])
        
        if i == 3:
            break
        
def check_loss_computation(data_loader, model, criterion_at, criterion_po):
    for i, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        at_annotations = batch['at_annotations'].to(device)
        po_annotations = batch['po_annotations'].to(device)
        
        aspect_logits, polarity_logits = model(input_ids, attention_mask=attention_mask)
        
        aspect_loss = criterion_at(aspect_logits.view(-1, 2), at_annotations.view(-1))
        polarity_loss = criterion_po(polarity_logits.view(-1, 4), po_annotations.view(-1))
        combined_loss = aspect_loss + polarity_loss
        
        print(f"Batch {i+1} - Aspect Loss: {aspect_loss.item()}, Polarity Loss: {polarity_loss.item()}, Combined Loss: {combined_loss.item()}")
        
        if i == 3:
            break
