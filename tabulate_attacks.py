import argparse
import pdb, traceback
import torch
import wandb
from copy import deepcopy
from attack import Classwise_membership_inference_attack, Combined_membership_inference_attack, Membership_inference_attack_instance 
from utils.models import load_model_defination
from utils.plot_utils import plot_ROC_curve
from utils.training_utils import get_device, load_model as load_saved_weights
from utils.datasets import DatasetWrapper
from utils.old_lib import record_JSON





def execute_attack(args, device,  target_dataset):

    target_model        = load_model_defination(args.target_model_name, target_dataset.num_channels, target_dataset.num_classes).to(device)
    load_saved_weights(target_model, filename = args.target_model_weights)
    attack_instance = Membership_inference_attack_instance( shadow_model_name   = args.shadow_model_name, 
                                                            shadow_count        = args.shadow_count, 
                                                            load_attack_dataset = args.load_attack_dataset,
                                                            save_attack_dataset = args.save_attack_dataset,
                                                            save_attack_model   = args.save_attack_model,
                                                            attack_model_name   = args.attack_model_name,
                                                            batchwise_loss      = args.batchwise_loss,
                                                            device              = device,
                                                            shadow_epochs       = args.num_shadow_epochs,
                                                            attack_epochs       = args.num_attack_epochs,
                                                            wandb_logging       = args.wandb_logging
                                                        )

    if args.combined_class:
        attack = Combined_membership_inference_attack(target_model, args.dataset_name, attack_instance, args.target_model_weights, args.wandb_logging)
    else:
        attack = Classwise_membership_inference_attack(target_model, args.dataset_name, attack_instance, args.target_model_weights, args.wandb_logging)
    
    return attack.start()

def tabulate_single_attack(args, device, batch_sizes, dataset_name, model_train_mode, model_name='efficientnet'):
    args.batchwise_loss = batch_sizes
    args.dataset_name = dataset_name
    target_dataset = DatasetWrapper(args.dataset_name)
    model_name=model_name
    model_train_mode = model_train_mode #0 for centralized, <n> for federated
    args.combined_class = True
    if model_train_mode == 0:
        args.target_model_weights = 'Centralizedefficientnet'+dataset_name
    else:
        args.target_model_weights = 'Federated'+str(model_train_mode)+'efficientnet'+args.dataset_name

    try:
        loss, accuracy, predictions = execute_attack(args, device,  target_dataset)
        plot_ROC_curve(predictions[0], predictions[1]) # type: ignore
    except:
        accuracy = 0
        traceback.print_exc()
        pdb.set_trace()
    print(f'\tAccuracy: {accuracy} for combined_class ={args.combined_class} with {args.target_model_weights} and batch_size {args.batchwise_loss}.')
   


def tabulate_all_attack(args, device, dataset_names, model_train_modes):
    accuracy_record = record_JSON()
    # accuracy_record.load()

    for dataset_name in dataset_names:
        
        args.dataset_name = dataset_name 
        for batch_size in ['single', 'batch_8', 'batch_16', 'batch', 'batch_64', 'batch_128', 'batch_256']:

            args.batchwise_loss = batch_size
            # for model_name in ['efficientnet', 'resnet', 'vgg', 'densenet', 'shufflenet', 'mobilenet']:
            model_name='efficientnet'

            # acc_client_wise = []

            for model_train_mode in model_train_modes:
                if model_train_mode == 0:
                    args.target_model_weights = 'Centralizedefficientnet'+dataset_name
                else:
                    args.target_model_weights = 'Federated'+str(model_train_mode)+'efficientnet'+dataset_name

                # acc_classWise = [0,0]
                combined_class = True
                # for combined_class in [True, False]:
                args.combined_class = combined_class
                target_dataset = DatasetWrapper(args.dataset_name)
                try:
                    mode = 'Combined Class' if args.combined_class else 'Classwise'
                    print(f'Executing {mode} Membership Inference Attack on {args.target_model_weights}')
                    # 
                    accuracy = 0.0,0

                    # accuracy  = accuracy_record.lookup( combined_class=combined_class, 
                    #             model_name = model_name, 
                    #             model_train_mode = model_train_mode, 
                    #             batch_size = batch_size, 
                    #             dataset_name = dataset_name)
                    
                    if accuracy[0] == 0.0:   # dummy vallue exists so overwrite
                        loss, accuracy, predictions = execute_attack(args, device,  target_dataset)
                        accuracy_record.record(accuracy, combined_class=combined_class,  model_name = model_name,  model_train_mode = model_train_mode,  batch_size = batch_size, dataset_name = dataset_name)
                        accuracy_record.save()
                        print(f'\tAccuracy: {accuracy} saved in record.')
                    else:
                        print(f'\tAccuracy: {accuracy} exists in record, skipping')
                        # pdb.set_trace()
                except:
                    traceback.print_exc()
                    print(traceback.print_exc())
                    print('\t\tThis run failed')
                    pdb.set_trace()


    accuracy_record.save()


def main():
    parser = argparse.ArgumentParser(description='A description of your program')
    parser.add_argument('-w', '--wandb_logging', action='store_true', help='Enable wandb logging')
    parser.add_argument('-e', '--num_shadow_epochs', type=int, default=100, help='Number of rounds of shadow training')
    parser.add_argument('-e1', '--num_attack_epochs', type=int, default=150, help='Number of rounds of attack training')
    parser.add_argument('-n', '--shadow_count', type=int, default=10, help='Number of shadow models')
    parser.add_argument('-a', '--attack_model_name', type=str, default= 'attack_classifier', help='Classifier for the attack model')
    parser.add_argument('-s', '--single', action='store_true', help='Check single attack')

    args = parser.parse_args()
    args.load_attack_dataset = True
    args.target_model_name = 'efficientnet'
    args.shadow_model_name = 'efficientnet'
    args.save_attack_dataset = False
    args.save_attack_model = False

    device = get_device()
    # pdb.set_trace()
    if args.single:
        tabulate_single_attack(args, device, 'batch_64','CIFAR10',2)
    else:
        # dataset_names =[ 'CIFAR10', 'CIFAR100','FashionMNIST', 'MNIST']
        train_modes =[0,2,3,5,10]
        dataset_names =[ 'CIFAR10']
        # train_modes =[2]
        tabulate_all_attack(args, device, dataset_names, train_modes)


    # 
    
    
    
    

if __name__ == '__main__':
    main()



