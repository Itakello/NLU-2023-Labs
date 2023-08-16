from functions import *
from utils import *
from model import *

if __name__ == "__main__":
    models = [(LM_LSTM_Dropout_SWA,
                {"emb_size": 250, "hidden_size": 400, "optimizer": torch.optim.AdamW, "lr": 0.0001}),
              (LM_LSTM_Weight_Tying,
                {"emb_size": 300, "hidden_size": 300, "optimizer": torch.optim.AdamW, "lr": 0.0001}),
            (LM_LSTM_Var_Dropout,
                {"emb_size": 250, "hidden_size": 400, "optimizer": torch.optim.SGD, "lr": 0.0001})
    ]
    
    train_raw, dev_raw, test_raw = get_raw_data()
    lang = Lang(train_raw + dev_raw + test_raw, ["<pad>", "<eos>"])
    vocab_len = len(lang.word2id)
    pad_id = lang.word2id["<pad>"]
    
    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum') 
    
    for model, hyp in tqdm(models, desc="Training models"):
        train_loader, dev_loader, test_loader = get_dataloaders(train_raw, dev_raw, test_raw, lang, pad_id, batch_size=256)
        model = model(hyp['emb_size'], hyp['hidden_size'], vocab_len, pad_id=pad_id).to(device)
        optimizer = hyp['optimizer'](model.parameters(), hyp['lr'])
        best_model, final_ppl = train_and_evaluate_avg(model, optimizer, criterion_train, criterion_eval, train_loader, dev_loader, test_loader)
        save_model(best_model, model.__class__.__name__, final_ppl)