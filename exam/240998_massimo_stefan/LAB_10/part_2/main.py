from functions import *
from model import *
from utils import *

if __name__ == "__main__":
    # Get the train dataloader
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side

    lr = 0.0001 # learning rate
    clip = 5 # Clip the gradient
    lang = get_lang()

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    
    model = BertForIntentAndSlotFilling(out_slot, out_int).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token
    train_loader, dev_loader, test_loader = get_dataloaders()
    sampled_epochs, losses_train, losses_dev = train_and_eval(model, optimizer, lang, train_loader, test_loader, dev_loader, criterion_slots, criterion_intents)
    plot_losses(sampled_epochs, losses_train, losses_dev)