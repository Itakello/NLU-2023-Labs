from functions import *
from utils import *
from model import *

if __name__ == "__main__":
    models = [
        (LM_LSTM_Adv,{"emb_size": 400, "hidden_size": 400, "optimizer": "AdamW", "lr": 0.0001})
    ]
    vocab_len = get_vocab_len()
    pad_index = get_pad_index()
    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_index)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_index, reduction='sum')
    for model, hyp in models:
        train_loader, dev_loader, test_loader = get_dataloaders()
        model = model(hyp['emb_size'], hyp['hidden_size'], vocab_len, pad_index=pad_index).to('cuda:0')
        model.apply(init_weights)
        optimizer = get_optimizer(hyp['optimizer'], model.parameters(), hyp['lr'])
        best_model, final_ppl = train_and_evaluate_avg(model, optimizer, criterion_train, criterion_eval, train_loader, dev_loader, test_loader, device='cuda:0')
        save_model(best_model, model.__class__.__name__, hyp['emb_size'], hyp['hidden_size'], hyp['optimizer'], hyp['lr'], final_ppl)