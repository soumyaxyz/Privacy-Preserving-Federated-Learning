from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import flwr as fl
import wandb
import torch
from utils.datasets import ContinuousDatasetWraper, load_partitioned_continous_datasets, load_partitioned_datasets, split_dataset
from utils.training_utils import make_private, save_model, wandb_init,  print_info, get_device, train, test, load_model as load_saved_weights
from utils.models import load_model_defination 
import argparse
import pdb,traceback
import cProfile



def transfer_learning(pretrained_model, device, wandb_logging=False,  source_dataset_name='SVHN', target_dataset_name='MNIST', model_name = 'basicCNN', differential_privacy=False):

    assert source_dataset_name != target_dataset_name

    [train_loaders_source, val_loaders_source, test_loader_source, _], num_channels_source, num_classes_source  = load_partitioned_datasets(num_clients=1, dataset_name=source_dataset_name)

    model = load_model_defination(model_name, num_channels=num_channels_source, num_classes=num_classes_source, differential_privacy=differential_privacy).to(device)
    load_saved_weights(model, filename =pretrained_model)
    
    val_loader_source = val_loaders_source[0]
    train_loader_scource = train_loaders_source [0]   

        
    loss, accuracy, _ = test(model, val_loader_source)
    tst_loss, tst_accuracy, _ = test(model, test_loader_source)
    trn_loss, trn_accuracy, _ = test(model, train_loader_scource)

    print(f"\n \n model details: {pretrained_model} \n")
    print(f"Final train set performance:\n\tloss {trn_loss}\n\taccuracy {trn_accuracy}")
    print(f"Final val set performance:\n\tloss {loss}\n\taccuracy {accuracy}")
    print(f"Final test set performance:\n\tloss {tst_loss}\n\taccuracy {tst_accuracy}")

    # # Freeze parameters of the pre-trained layers
    # for param in model.parameters():
    #     param.requires_grad = False

    # # Unfreeze parameters of the final classification layer
    # for param in model.fc3.parameters():
    #     param.requires_grad = True
    


    [train_loaders_target, val_loaders_target, test_loader_target, _], num_channels_target, num_classes_target  = load_partitioned_datasets(num_clients=1, dataset_name=target_dataset_name)
    val_loader_target = val_loaders_target[0]
    train_loader_target = train_loaders_target [0]   

    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    # if num_classes_source != num_classes_target:
    #     replace_classifying_layer(model, classes)

    loss, accuracy, _ = test(model, test_loader_target)

    print(f"\n \n model details: {pretrained_model} \n")
    print(f"Final val set performance:\n\tloss {loss}\n\taccuracy {accuracy}")

    optimizer = torch.optim.Adam(model.parameters())
    epoch = 5

    model, optimizer, train_loader_target = make_private(differential_privacy, model, optimizer, train_loader_target)

    model, optimizer, val_loss, val_accuracy, _  = train(model, train_loader_target, val_loader_target, epoch, optimizer, verbose=False, wandb_logging=wandb_logging)
    
    loss, accuracy, _ = test(model, test_loader_target)


    print(f"\n \n model details: {pretrained_model} \n")
    print(f"Final val set performance:\n\tloss {loss}\n\taccuracy {accuracy}")




def continous_learning(device, epochs=50, wandb_logging=False,  dataset_name='continous_SVHN', model_name = 'efficientnet', patience=2, differential_privacy=False):
    

    print_info(device, model_name, dataset_name)  

    continous_datasets = ContinuousDatasetWraper(dataset_name)
    saved_model_names = []

    _, _, num_channels, num_classes = continous_datasets.splits[0]

    model = load_model_defination(model_name, num_channels, num_classes, differential_privacy=differential_privacy).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    for i,dataset_split in enumerate(continous_datasets.splits):
        [train_loaders, val_loaders, test_loader, _ ], num_channels_i, num_classes_i = load_partitioned_continous_datasets(num_clients=1, dataset_split=dataset_split)
        assert num_channels_i == num_channels
        assert num_classes_i == num_classes

        train_loader = train_loaders[0]
        val_loader = val_loaders[0]        
        

        comment = 'Centralized_'+model_name+'_'+dataset_name+'_'+str(i)
        if wandb_logging:
            wandb_init(comment=comment, model_name=model_name, dataset_name=dataset_name)
            wandb.watch(model, log_freq=100)
            
        model, optimizer, train_loader = make_private(differential_privacy, model, optimizer, train_loader)

        model, optimizer, val_loss, val_accuracy, _  = train(model, train_loader, val_loader, epochs, optimizer, verbose=False, wandb_logging=wandb_logging, patience=patience)
        loss, accuracy, _ = test(model, test_loader)

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

        save_model(model, optimizer, savefilename)

        saved_model_names.append(savefilename)

    return saved_model_names






def main():
    parser = argparse.ArgumentParser(description='A description of your program')
    parser.add_argument('-r', '--num_experiments', type=int, default=1, help='Number of experiments')
    parser.add_argument('-e', '--num_epochs', type=int, default=50, help='Number of rounds')
    parser.add_argument('-s', '--save_filename', type=str, default=None, help='Save filename')
    parser.add_argument('-w', '--wandb_logging', action='store_true', help='Enable wandb logging')
    parser.add_argument('-d', '--dataset_name', type=str, default='continous_SVHN', help='Dataset name')
    parser.add_argument('-sd', '--source_dataset_name', type=str, default='SVHN', help='Source_dataset name')
    parser.add_argument('-p', '--patience', type=int, default=5, help='Patience')
    parser.add_argument('-td', '--target_dataset_name', type=str, default='MNIST', help='Target_dataset name')
    parser.add_argument('-m', '--model_name', type=str, default='efficientnet', help='Model name')
    parser.add_argument('-pm', '--pretrained_model', type=str, default= None, help='if provided, evaluate transfer learning on this saved model')
    parser.add_argument('-dp', '--differential_privacy', action='store_true', help='Enable differential privacy')
    args = parser.parse_args()

    device = get_device()

    if args.pretrained_model:
        transfer_learning(args.pretrained_model, device, args.wandb_logging,  args.source_dataset_name, args.target_dataset_name, args.model_name, args.differential_privacy)
    else:
        for _ in range(args.num_experiments):
            saved_model_names = continous_learning(device, args.num_epochs, args.wandb_logging,  args.dataset_name, args.model_name, args.patience, args.differential_privacy)
            print(f'{saved_model_names=}')
        

if __name__ == "__main__":  
    cProfile.run('main()', 'output.prof')  