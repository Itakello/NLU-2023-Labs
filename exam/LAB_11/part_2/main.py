from functions import *
from model import *
from utils import *
from torch.nn import CrossEntropyLoss
from torch import optim

if __name__ == "__main__":
    lang = Lang()
    
    train_loader, test_loader, dev_loader = get_dataloaders(lang)
    
    criterion_at = CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_po = CrossEntropyLoss(ignore_index=PAD_TOKEN)
    
    model = JointABSA(num_aspect_labels=3, num_polarity_labels=5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    model = train_and_eval(model, optimizer, train_loader, test_loader, dev_loader, criterion_at, criterion_po, lang)
    save_model(model, 'JointABSA')