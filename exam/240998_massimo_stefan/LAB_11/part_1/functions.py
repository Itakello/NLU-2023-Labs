import torch
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import numpy as np

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train_loop(data, optimizer, criterion, model):
    model.train()
    total_loss = 0
    for batch in data:
        optimizer.zero_grad() # Zeroing the gradient
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        logits = model(input_ids, attention_mask=attention_masks, labels=labels).logits
        loss = criterion(logits, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_training_loss = total_loss / len(data)
    return avg_training_loss

# TODO check
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# TODO check
def eval_loop(data, criterion, model):
    model.eval()
    total_loss = 0
    total_eval_accuracy = 0
    for batch in data:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            total_loss += loss.item()
            logits = outputs[1]
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
    avg_val_loss = total_loss / len(data)
    avg_val_accuracy = total_eval_accuracy / len(data)
    return avg_val_loss, avg_val_accuracy

def train_and_eval(model, optimizer, criterion, train_loader, test_loader, dev_loader):
    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    for x in tqdm(range(1,n_epochs)):
        loss = train_loop(train_loader, optimizer, criterion, model)
        if x % 5 == 0:
            # TODO fix patience and results
            sampled_epochs.append(x)
            losses_train.append(loss)
            loss_dev, _ = eval_loop(dev_loader, criterion, model)
            losses_dev.append(loss_dev)

    _, accuracy = eval_loop(test_loader, criterion, model)
    print('Intent Accuracy:', accuracy)
    return sampled_epochs, losses_train, losses_dev
    
def plot_losses(sampled_epochs, losses_train, losses_dev):
    plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
    plt.title('Train and Dev Losses')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(sampled_epochs, losses_train, label='Train loss')
    plt.plot(sampled_epochs, losses_dev, label='Dev loss')
    plt.legend()
    plt.savefig('test.png')