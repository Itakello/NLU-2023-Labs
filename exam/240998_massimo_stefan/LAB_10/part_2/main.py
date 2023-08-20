from functions import *
from model import *
from utils import *
from torch.optim import AdamW

if __name__ == "__main__":

    LEARNING_RATE = 0.0001
    
    # Loading raw data
    train_raw, dev_raw, test_raw = get_raw_data()
    
    lang = Lang(train_raw, dev_raw, test_raw)
    
    model = BertForIntentAndSlotFilling(len(lang.slot2id), len(lang.intent2id)).to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token
    
    # Getting data loaders
    train_loader, dev_loader, test_loader = get_dataloaders(train_raw, dev_raw, test_raw, lang, tokenizer)
    
    model, performance_metric = train_and_eval(model, optimizer, lang, train_loader, dev_loader, test_loader, criterion_slots, criterion_intents, tokenizer)
    save_model(model, model.__class__.__name__, performance_metric)