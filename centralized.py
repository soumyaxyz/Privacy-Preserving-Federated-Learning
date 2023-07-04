from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import flwr as fl
import wandb
import torch
from utils.client_utils import load_partitioned_datasets
from utils.training_utils import save_model, wandb_init,  print_info, get_device, train, test
from utils.models import load_model 
import argparse



def train_centralized(epochs=50, device="cpu", wandb_logging=True, savefilename=None, dataset_name='CIFAR10'):
    model = load_model("basic_CNN", num_channels=3, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    model_name=model.__class__.__name__


    print_info(device, model_name, dataset_name)    

    train_loaders, val_loaders, test_loader, _ = load_partitioned_datasets(num_clients=1, dataset_name=dataset_name)

    train_loader = train_loaders[0]
    val_loader = val_loaders[0]   
        
    

    comment = 'Centralized_item_'+model_name+'_'+dataset_name
    if wandb_logging:
        wandb_init(comment=comment, model_name=model_name, dataset_name=dataset_name)

    model, optimizer = train(model, train_loader, val_loader, epochs, optimizer, verbose=False, wandb_logging=wandb_logging)
    loss, accuracy = test(model, test_loader)

    if wandb_logging:
        wandb.log({"test_acc": accuracy, "test_loss": loss})
        wandb.finish()
    print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")
          
    
    if not savefilename:
        savefilename = comment

    save_model(model, optimizer, savefilename)
     


def main():
    parser = argparse.ArgumentParser(description='A description of your program')
    parser.add_argument('-r', '--num_experiments', type=int, default=1, help='Number of experiments')
    parser.add_argument('-e', '--num_epochs', type=int, default=50, help='Number of rounds')
    parser.add_argument('-s', '--save_filename', type=str, default=None, help='Save filename')
    parser.add_argument('-w', '--wandb_logging', action='store_true', help='Enable wandb logging')
    parser.add_argument('-d', '--dataset_name', type=str, default='CIFAR10', help='Dataset name')
    args = parser.parse_args()

    device = get_device()

    for _ in range(args.num_experiments):
        train_centralized(args.num_epochs, device, args.wandb_logging, args.save_filename, args.dataset_name)
        



if __name__ == "__main__":       
    main()
    