from functions import *
from model import *
from utils import *

if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    lr = 0.0001 # learning rate    
    
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    sentences, labels, encoder = get_sub_data()
    train_dataloader, eval_dataloader, test_dataloader = get_sub_dataloaders(sentences, labels)
    sampled_epochs, losses_train, losses_dev = train_and_eval(model, optimizer, criterion, train_dataloader, test_dataloader, eval_dataloader)
    plot_losses(sampled_epochs, losses_train, losses_dev)