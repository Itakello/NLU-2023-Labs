import torch
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedKFold
from utils import create_loader
from model import *
import os

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def save_model(model, model_name):
    path = 'model_bin/' + model_name + '.pt'
    file_path = os.path.join(os.path.dirname(__file__), path)
    torch.save(model.state_dict(), file_path)

def load_model(model_name):
    path = 'model_bin/' + model_name + '.pt'
    file_path = os.path.join(os.path.dirname(__file__), path)
    model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels = 2,
            output_attentions = False,
            output_hidden_states = False
        ).to(device)
    model.load_state_dict(torch.load(file_path))
    return model

def train_loop(data, optimizer, criterion, model):
    model.train()
    total_loss = 0
    for batch in tqdm(data, desc='Training', leave=False):
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

def eval_loop(data, criterion, model):
    model.eval()
    total_loss = 0
    total_eval_accuracy = 0
    total_eval_examples = 0
    for batch in tqdm(data, desc='Evaluating', leave=False):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            # Forward pass
            logits = model(input_ids, attention_mask=attention_mask, labels=labels).logits
            # Compute loss
            loss = criterion(logits, labels)
            total_loss += loss.item()
            # Compute accuracy
            preds = torch.argmax(logits, dim=1)
            correct_preds = (preds == labels).sum().item()
            total_eval_accuracy += correct_preds
            total_eval_examples += labels.size(0)
    avg_val_loss = total_loss / len(data)
    avg_val_accuracy = total_eval_accuracy / total_eval_examples
    return avg_val_loss, avg_val_accuracy

def k_fold_evaluation(criterion, tokenizer, sentences, labels, model_name, n_splits=2):
    skf = StratifiedKFold(n_splits=n_splits)
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_accuracy = 0

    for train_index, val_index in skf.split(sentences, labels):
        # Split data into train and validation sets
        X_train, X_val = [sentences[i] for i in train_index], [sentences[i] for i in val_index]
        y_train, y_val = [labels[i] for i in train_index], [labels[i] for i in val_index]

        train_loader = create_loader(X_train, y_train, tokenizer, shuffle=True)
        val_loader = create_loader(X_val, y_val, tokenizer)
        
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels = 2,
            output_attentions = False,
            output_hidden_states = False
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.0001)

        # Train the model
        train_loss = train_loop(train_loader, optimizer, criterion, model)

        # Evaluate the model
        val_loss, val_accuracy = eval_loop(val_loader, criterion, model)
        
        # Check if this model has better validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(model, model_name)

        # Store results
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    # Compute average results
    avg_train_loss = sum(train_losses) / n_splits
    avg_val_loss = sum(val_losses) / n_splits
    avg_val_accuracy = sum(val_accuracies) / n_splits

    print(f"Performances {model_name} on the subjectivity task")
    print("Train loss:", avg_train_loss)
    print("Validation loss:", avg_val_loss)
    print("Validation accuracy:", avg_val_accuracy)
    
def filter_subj_doc(sentences, tokenizer, old_model_name):
    # TODO fix so that it sent_tokenize the documents, it performs the classification and then it joins the sentences
    loader = create_loader(sentences, ['obj'] * len(sentences), tokenizer) # NOTE dummy labels
    model = load_model(old_model_name)
    model.eval()
    filtered_sentences = []
    filtered_labels = []
    for batch in tqdm(loader, desc='Filtering subj sentences', leave=False):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # Forward pass
            logits = model(input_ids, attention_mask=attention_mask).logits
            # Compute accuracy
            preds = torch.argmax(logits, dim=1)
            # take only the sentences that are classified as subjective
            subj_indices = (preds == 1).nonzero(as_tuple=True)[0]
            subj_sentences = [batch['sentence_text'][i] for i in subj_indices]
            filtered_sentences.extend(subj_sentences)
            subj_labels = [batch['label'][i] for i in subj_indices]
            filtered_labels.extend(subj_labels)
        torch.cuda.empty_cache()
    return filtered_sentences, filtered_labels