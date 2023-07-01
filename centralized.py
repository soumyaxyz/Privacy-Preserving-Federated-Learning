from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import flwr as fl
import wandb
import torch
from utils.client_utils import load_datasets
from utils.training_utils import save_model, wandb_init,  print_info, get_device, train, test
from utils.models import basicCNN as Net
import argparse



def train_centralized(epochs=50, device="cpu", wandb_logging=True, save_location="./saved_models/"):
    train_loaders, val_loaders, test_loader, _ = load_datasets(num_clients=1)

    train_loader = train_loaders[0]
    val_loader = val_loaders[0]
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    comment = 'Centraized_'+model.__class__.__name__
    if wandb_logging:
        wandb_init(comment=comment, model_name=model_name )

    model, optimizer = train(model, train_loader, val_loader, epochs, optimizer, verbose=False, wandb_logging=wandb_logging)
    loss, accuracy = test(model, test_loader)

    if wandb_logging:
        wandb.log({"test_acc": accuracy, "test_loss": loss})
        wandb.finish()
    print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")
          
    
    save_path = save_location + comment + ".pt"

    save_model(model, optimizer, save_path)
     


def main():
    parser = argparse.ArgumentParser(description='A description of your program')
    parser.add_argument('-r', '--num_experiments', type=int, default=1, help='Number of experiments')
    parser.add_argument('-e', '--num_epochs', type=int, default=50, help='Number of rounds')
    parser.add_argument('-s', '--save_location', type=str, default='./saved_models/', help='Save location')
    parser.add_argument('-w', '--wandb_logging', action='store_true', help='Enable wandb logging')
    args = parser.parse_args()

    device = get_device()
    print_info(device)

    for _ in range(args.num_experiments):
        train_centralized(args.num_epochs, device, args.wandb_logging, args.save_location)
        



if __name__ == "__main__":       
    main()
    