from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import flwr as fl
import wandb
import torch
from torch import nn 
from utilities.datasets import IncrementalDatasetWraper, load_partitioned_datasets, get_dataloaders_subset
from utilities.training_utils import Trainer, save_model, wandb_init,  print_info, get_device, train, test, load_model as load_saved_weights
from utilities.models import load_model_defination 
import argparse
import pdb,traceback
import cProfile


def incremental_learning(device, epochs=50, wandb_logging=False,  dataset_name='continous_SVHN', model_name = 'efficientnet', patience=2):
    

    print_info(device, model_name, dataset_name)  

    continous_datasets = IncrementalDatasetWraper(dataset_name)
    saved_model_names = []

    _, _, num_channels, num_classes = continous_datasets.splits[0]

    model = load_model_defination(model_name, num_channels, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for i,dataset_split in enumerate(continous_datasets.splits):
        [train_loaders, val_loaders, test_loader, _ ], num_channels_i, num_classes_i = load_partitioned_datasets(num_clients=1, dataset_name=dataset_name, split=i)
        assert num_channels_i == num_channels
        assert num_classes_i == num_classes

        train_loader = train_loaders[0]
        val_loader = val_loaders[0]        
        

        comment = 'Centralized_'+model_name+'_'+dataset_name+'_'+str(i)
        if wandb_logging:
            wandb_init(comment=comment, model_name=model_name, dataset_name=dataset_name)
            wandb.watch(model, log_freq=100)
        
        model_information = Trainer(model, train_loader, val_loader, test_loader, optimizer, criterion,device = device, is_binary=False,summary_writer=None)
        model_information, val_loss, val_accuracy, _  = train(model_information, epochs, verbose=True, wandb_logging=False, round_no=None)
        
        #model, optimizer, val_loss, val_accuracy, _  = train(model, train_loader, val_loader, epochs, optimizer, verbose=False, wandb_logging=wandb_logging, patience=patience)
        loss, accuracy, _ = test(model_information, test_loader)

        if wandb_logging:
            wandb.log({"test_acc": accuracy, "test_loss": loss})
            wandb.finish()
        print(f"Final validation set performance:\n\tloss {val_loss}\n\taccuracy {val_accuracy}")
        print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")
        print('\n')
        # Print a line of dashes
        print('-' * 20)
        print('\n')
            
        
        savefilename = comment

        save_model(model_information, optimizer, savefilename)

        saved_model_names.append(savefilename)

    return saved_model_names


def evaluate(evaluation_model, device, wandb_logging=True,  dataset_name='CIFAR10', model_name = 'efficientnet', differential_privacy=False):
    
    print_info(device, model_name, dataset_name, eval=True)    
    try:

        if  model_name == 'lgb':
            pass
            
        else:
            
            dataset_name, index = dataset_name.split('-')
            target_dataset = IncrementalDatasetWraper(dataset_name, audit_mode=False)
            target_dataset.select_split(int(index))

            [train_loaders, val_loaders, test_loader, _ ], num_channels, num_classes = load_partitioned_datasets(num_clients=1, split=target_dataset.data_split)

            val_loader = val_loaders[0]   
            train_loader = train_loaders[0]

            test_loader_size = len(test_loader.dataset)


            train_loader = get_dataloaders_subset(train_loader, test_loader_size)
            
           
            model = load_model_defination(model_name, num_channels, num_classes, differential_privacy).to(device) 
            optimizer = torch.optim.Adam(model.parameters())


            load_saved_weights(model, filename =evaluation_model)

            

            comment = 'Test_Centralized_('+evaluation_model+')_'+model_name+'_'+dataset_name
            if wandb_logging:
                wandb_init(comment=comment, model_name=model_name, dataset_name=dataset_name)
                wandb.watch(model, log_freq=100)
                
            trn_loss, trn_accuracy, predA = test(model, train_loader)
            val_loss, val_accuracy, _ = test(model, val_loader)
            tst_loss, tst_accuracy, predB = test(model, test_loader)

            print(f"Final training set performance:\n\tloss {trn_loss}\n\taccuracy {trn_accuracy}")

            if wandb_logging:
                wandb.log({"train_acc": trn_accuracy, "train_loss": trn_loss})
                wandb.log({"acc": val_accuracy, "loss": val_loss}, step = 100)
                wandb.log({"test_acc": tst_accuracy, "test_loss": tst_loss})
                wandb.finish()
            print(f"Final validation set performance:\n\tloss {val_loss}\n\taccuracy {val_accuracy}")
            print(f"Final test set performance:\n\tloss {tst_loss}\n\taccuracy {tst_accuracy}")

            if wandb_logging:
                wandb.finish()
    except Exception as e:
        traceback.print_exc()
        pdb.set_trace()



def main():
    parser = argparse.ArgumentParser(description='A description of your program')
    parser.add_argument('-r', '--num_experiments', type=int, default=1, help='Number of experiments')
    parser.add_argument('-e', '--num_epochs', type=int, default=50, help='Number of rounds')
    parser.add_argument('-s', '--save_filename', type=str, default=None, help='Save filename')
    parser.add_argument('-w', '--wandb_logging', action='store_true', help='Enable wandb logging')
    parser.add_argument('-d', '--dataset_name', type=str, default='incremental_SVHN', help='Dataset name')
    parser.add_argument('-sd', '--source_dataset_name', type=str, default='SVHN', help='Source_dataset name')
    parser.add_argument('-p', '--patience', type=int, default=5, help='Patience')
    parser.add_argument('-td', '--target_dataset_name', type=str, default='MNIST', help='Target_dataset name')
    parser.add_argument('-m', '--model_name', type=str, default='efficientnet', help='Model name')
    parser.add_argument('-em', '--evaluation_model', type=str, default= None, help='if provided, evaluate on this saved model')
    parser.add_argument('-pm', '--pretrained_model', type=str, default= None, help='if provided, evaluate transfer learning on this saved model')
    args = parser.parse_args()

    device = get_device()
    
    if args.evaluation_model:
        evaluate(args.evaluation_model, device, args.wandb_logging, args.dataset_name, args.model_name, args.differential_privacy)
    else:
        for _ in range(args.num_experiments):
            saved_model_names = incremental_learning(device, args.num_epochs, args.wandb_logging,  args.dataset_name, args.model_name, args.patience)
            print(f'{saved_model_names}')  

if __name__ == "__main__":  
    cProfile.run('main()', 'output.prof')  