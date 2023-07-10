from functions import *
from model import *
from utils import *

if __name__ == "__main__":
    
    device = 'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
    hid_size = 200
    emb_size = 300

    lr = 0.0001 # learning rate
    clip = 5 # Clip the gradient
    lang = get_lang()

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token
    train_loader, dev_loader, test_loader = get_dataloaders()
    sampled_epochs, losses_train, losses_dev = train_and_eval(model, optimizer, lang, train_loader, test_loader, dev_loader, criterion_slots, criterion_intents)
    plot_losses(sampled_epochs, losses_train, losses_dev)