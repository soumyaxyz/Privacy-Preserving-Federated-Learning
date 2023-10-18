from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import flwr as fl
import wandb
import torch
from utils.datasets import load_partitioned_datasets
from utils.training_utils import save_model, wandb_init,  print_info, get_device, train, test, load_model as load_saved_weights
from utils.models import load_model_defination 
import argparse
import pdb,traceback

def evaluate(evaluation_model, device, wandb_logging=True,  dataset_name='CIFAR10', model_name = 'basic_CNN'):
    


    print_info(device, model_name, dataset_name)    

    [_, val_loaders, test_loader, _], num_channels, num_classes = load_partitioned_datasets(num_clients=1, dataset_name=dataset_name)

    
    val_loader = val_loaders[0]   


    # print(f"Training on {model_name} with {dataset_name} in {device} using PyTorch {torch.__version__} and Flower {fl.__version__}")
    model = load_model_defination(model_name, num_channels, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    load_saved_weights(model, filename =evaluation_model)
        
    

    comment = 'Test_Centralized_('+evaluation_model+')_'+model_name+'_'+dataset_name
    if wandb_logging:
        wandb_init(comment=comment, model_name=model_name, dataset_name=dataset_name)
        wandb.watch(model, log_freq=100)
        

    loss, accuracy = test(model, val_loader)
    tst_loss, tst_accuracy = test(model, test_loader)

    if wandb_logging:
        wandb.log({"acc": accuracy, "loss": loss}, step = 100)
        wandb.log({"test_acc": tst_accuracy, "test_loss": tst_loss})
        wandb.finish()
    print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")
          
    if wandb_logging:
        wandb.finish()
    # pdb.set_trace()


def train_centralized(epochs, device, wandb_logging=True, savefilename=None, dataset_name='CIFAR10', model_name = 'basic_CNN'):


    [train_loaders, val_loaders, test_loader, _ ], num_channels, num_classes = load_partitioned_datasets(num_clients=1, dataset_name=dataset_name) 


    # print(f"Training on {model_name} with {dataset_name} in {device} using PyTorch {torch.__version__} and Flower {fl.__version__}")
    model = load_model_defination(model_name, num_channels, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters())


    print_info(device, model_name, dataset_name)    

    

    train_loader = train_loaders[0]
    val_loader = val_loaders[0]   
        
    

    comment = 'Centralized_'+model_name+'_'+dataset_name
    if wandb_logging:
        wandb_init(comment=comment, model_name=model_name, dataset_name=dataset_name)
        wandb.watch(model, log_freq=100)
        

    model, optimizer, val_loss, val_accuracy  = train(model, train_loader, val_loader, epochs, optimizer, verbose=False, wandb_logging=wandb_logging)
    loss, accuracy = test(model, test_loader)

    if wandb_logging:
        wandb.log({"test_acc": accuracy, "test_loss": loss})
        wandb.finish()
    print(f"Final validation set performance:\n\tloss {val_loss}\n\taccuracy {val_accuracy}")
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
    parser.add_argument('-m', '--model_name', type=str, default='basicCNN', help='Model name')
    parser.add_argument('-em', '--evaluation_model', type=str, default= None, help='if provided, evaluate on this saved model')
    args = parser.parse_args()

    device = get_device()
    if args.evaluation_model:
        evaluate(args.evaluation_model, device, args.wandb_logging, args.dataset_name, args.model_name)
    else:
        for _ in range(args.num_experiments):
            train_centralized(args.num_epochs, device, args.wandb_logging, args.save_filename, args.dataset_name, args.model_name)
        



if __name__ == "__main__":       
    main()
    