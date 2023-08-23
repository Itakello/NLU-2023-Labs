from functions import *
from model import *
from utils import *
from torch.optim import AdamW

if __name__ == "__main__":
    # Constants
    HID_SIZE = 200
    EMB_SIZE = 300
    LEARNING_RATE = 0.0001

    # Loading raw data
    train_raw, dev_raw, test_raw = get_raw_data()

    # Getting language object containing vocab and slot/intent details
    lang = get_lang(train_raw, dev_raw, test_raw)

    # Model Parameters
    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    # Initializing Model and moving it to device
    model = ModelIAS(HID_SIZE, out_slot, out_int, EMB_SIZE, vocab_len, pad_index=PAD_TOKEN).to(device)

    # Setting up optimizer and loss functions
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()  # We do not have the pad token

    # Getting data loaders
    train_loader, dev_loader, test_loader = get_dataloaders(train_raw, dev_raw, test_raw, lang)

    model, performance_metric = train_and_eval(model, optimizer, lang, train_loader, dev_loader, test_loader, criterion_slots, criterion_intents)
    save_model(model, model.__class__.__name__, performance_metric)