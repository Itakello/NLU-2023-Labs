from functions import *
from model import *
from utils import *
from torch.nn import CrossEntropyLoss
from torch import optim

if __name__ == "__main__":
    train_loader, test_loader, dev_loader = get_dataloaders(32, 100)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
    
    criterion_at = CrossEntropyLoss()
    criterion_po = CrossEntropyLoss(ignore_index=0)  # ignore the padding index -1
    model = JointABSA(num_aspect_labels=2, num_polarity_labels=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    #check_model_predictions(dev_loader, model)
    #check_loss_computation(dev_loader, model, criterion_at, criterion_po)
    sampled_epochs, losses_train, losses_dev = train_and_eval(model, optimizer, train_loader, test_loader, dev_loader, criterion_at, criterion_po)