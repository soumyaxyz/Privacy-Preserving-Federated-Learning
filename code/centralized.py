import wandb
import torch
import argparse
import pdb,traceback
import numpy as np

from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10


from utilities.training_utils import Trainer, save_model, wandb_init,  print_info, get_device, train, test, load_model as load_saved_weights
from utilities.models import load_model_defination
from utilities.datasets import load_partitioned_datasets





def check_dataset(dataset_name ): 
    if "incremental" in dataset_name:
        try:
            dataset_name, num_splits = dataset_name.split("_")
            splits=[*range(int(num_splits))]
        except:
            splits=[*range(4)]
    else:
        splits=None
    
    
    return dataset_name, splits

    
def setup(device, wandb_logging=True, dataset_name='CIFAR10', model_name = 'basic_CNN', differential_privacy=False, split=None):

    print_info(device, model_name, dataset_name)
    data_index = 0   # centralized 

    [trainloader, valloaders, testloader, _ ], num_channels, num_classes = load_partitioned_datasets(num_clients=1, dataset_name=dataset_name, 
                                                                                                        data_path="~/dataset", batch_size=32, split=split) 

    train_loader = trainloader[data_index]
    valloader = valloaders[data_index]
    

    comment = 'Centralized_'+model_name+'_'+dataset_name

    if split is not None:
        comment = comment + f'_{split}'
        # print(f'comment: {comment}')
    # else:
    #     pdb.set_trace()

    if wandb_logging:
        wandb_init(comment=comment, model_name=model_name, dataset_name=dataset_name)

    model = load_model_defination(model_name, num_channels, num_classes) 
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)



    model_information = Trainer(model, 
                                train_loader, 
                                valloader, 
                                testloader, 
                                optimizer, 
                                criterion,
                                device = device, 
                                is_binary=False, 
                                summary_writer=None)
    
    return model_information, comment
   

def train_centralized(model_information, epochs,  wandb_logging=True, savefilename="unnamed_centrailzed"):
    print(f"Training centralized model for {epochs} epochs")
    

    model_information, val_loss, val_accuracy, _  = train(model_information, epochs, verbose=False, wandb_logging=False, round_no=None)  


    if wandb_logging:
        wandb.watch(model_information.model, log_freq=100)
        

    loss, accuracy, _ = test(model_information)

    if wandb_logging:
        wandb.log({"val_acc": val_accuracy, "val_loss": val_loss})
        wandb.log({"test_acc": accuracy, "test_loss": loss})
        wandb.finish()


    print(f"Final validation set performance:\n\tloss {val_loss}\n\taccuracy {val_accuracy}")
    print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")          

    save_model(model_information.model, model_information.optimizer, savefilename, print_info=True)

    return model_information


def evaluate(model_information, evaluation_model, wandb_logging=True):

    load_saved_weights(model_information.model, model_information.optimizer, evaluation_model)
    

    # model_informationTrain = model_information.from_trainer(model_information, testloader= model_information.Trainloader)
    # model_informationVal = model_information.from_trainer(model_information, testloader= model_information.Valloader)
    
    
    trn_loss, ten_accuracy, _ = test(model_information, mode='train')
    val_loss, val_accuracy, _ = test(model_information,  mode='val')
    tst_loss, tst_accuracy, _ = test(model_information)

    try:

        if wandb_logging:
            wandb.log({"train_acc": ten_accuracy, "train_loss": trn_loss})
            wandb.log({"acc": val_accuracy, "loss": val_loss}, step = 100)
            wandb.log({"test_acc": tst_accuracy, "test_loss": tst_loss})
            wandb.finish()
        print(f"Final validation set performance:\n\tloss {val_loss}\n\taccuracy {val_accuracy}")
        print(f"Final test set performance:\n\tloss {tst_loss}\n\taccuracy {tst_accuracy}")

        # plot_histogram(predA, predB)
            
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
    parser.add_argument('-d', '--dataset_name', type=str, default='CIFAR10', help='Dataset name')
    parser.add_argument('-m', '--model_name', type=str, default='basicCNN', help='Model name')
    parser.add_argument('-em', '--evaluation_model', type=str, default= None, help='if provided, evaluate on this saved model')
    parser.add_argument('-dp', '--differential_privacy', action='store_true', help='Enable differential privacy')
    args = parser.parse_args()

    device = get_device()

 
    dataset_name, splits = check_dataset(args.dataset_name)


    if splits is not None:   #incremental mode
        model_information = None

        for split in splits:
            model_information, comment = setup( device, args.wandb_logging, dataset_name, args.model_name, args.differential_privacy, split)           

            save_filename = comment
            # print(f"save_filename: {save_filename}")

            # if args.evaluation_model:
            #     evaluate(model_information, args.evaluation_model, args.wandb_logging)
            model_information = train_centralized(model_information, args.num_epochs, args.wandb_logging, save_filename)


    else: #standard mode

        model_information, comment = setup( device, args.wandb_logging, dataset_name, args.model_name, args.differential_privacy)

        if not args.save_filename:
            args.save_filename = comment

        if args.evaluation_model:
            evaluate(model_information, args.evaluation_model, args.wandb_logging)
        else:
            for _ in range(args.num_experiments):
                train_centralized(model_information, args.num_epochs, args.wandb_logging, args.save_filename)
        



if __name__ == "__main__":       
    main()
    