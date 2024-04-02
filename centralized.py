from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import flwr as fl
import wandb
import torch
from utils.datasets import load_partitioned_datasets, get_dataloaders_subset
from utils.training_utils import make_private, save_model, wandb_init,  print_info, get_device, train, test, load_model as load_saved_weights
from utils.models import Load_LGB, load_model_defination 
import argparse
import matplotlib.pyplot as plt
import pdb,traceback




def get_confidence(prediction, filtered = False, value=1):
    (confidence, eval_results) = prediction # type: ignore   
    if filtered:
        filtered_confidence = confidence[eval_results == value]
        confidence = filtered_confidence 

    # confidence = torch.nn.functional.softmax(confidence, dim=1)
    # pdb.set_trace()

    confidence = sorted(confidence)
    return confidence

def plot_histogram(predTrain, predTest):
    dpi_value = 100
    fig, axs = plt.subplots(1, 3, figsize=(15, 3), dpi=dpi_value)  # 1 row, 3 columns
    title = ['All', 'Correctly Classified', 'Incorrectly Classified']

    # Set a common range for the x-axis
    x_axis_range = (0, 1)

    # Initialize variables to track the maximum y-axis limit
    max_y_limit = 0

    for i, ax in enumerate(axs):
        if i == 0:
            trn_conf = get_confidence(predTrain, filtered=False, value=1)
            tst_conf = get_confidence(predTest, filtered=False, value=1)
        elif i == 1:
            trn_conf = get_confidence(predTrain, filtered=True, value=1)
            tst_conf = get_confidence(predTest, filtered=True, value=1)
        else:
            trn_conf = get_confidence(predTrain, filtered=True, value=0)
            tst_conf = get_confidence(predTest, filtered=True, value=0)

        # ax.title.set_text(title[i])

        # Update the maximum y-axis limit
        max_y_limit = max(max_y_limit, max(ax.hist(trn_conf, bins=20, range=x_axis_range, alpha=0.5, label='train', edgecolor='black')[0]))

        max_y_limit = max(max_y_limit, max(ax.hist(tst_conf, bins=20, range=x_axis_range, alpha=0.5, label='test', edgecolor='black')[0]))
        
        ax.set_xlabel(title[i], fontsize=18)
        ax.set_ylabel('Sample count', fontsize=18)
        ax.legend(fontsize=18)

    # Set the same y-axis limits for all subplots
    for ax in axs:
        ax.set_ylim(0, max_y_limit*1.05)

    plt.tight_layout()
    plt.show()




def evaluate(evaluation_model, device, wandb_logging=True,  dataset_name='CIFAR10', model_name = 'efficientnet', differential_privacy=False):
    


    print_info(device, model_name, dataset_name, eval=True)    
    try:

        if  model_name == 'lgb':
            import utils.datasets as d 

            from torch.utils.data import DataLoader, TensorDataset
            from sklearn.model_selection import train_test_split
            data_splits = d.load_incremental_Microsoft_Malware()


            train_subset = data_splits[0][0]
            X_train = train_subset.dataset.tensors[0].numpy()  # Assuming features are tensor[0]
            Y_train = train_subset.dataset.tensors[1].numpy()  # Assuming labels are tensor[1]

            # Splitting the train subset into train and validation sets
            X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)


            comment = model_name+'_Centralized_'+dataset_name

            if wandb_logging:
                wandb_init(comment=comment, model_name=model_name, dataset_name=dataset_name)

            LGB = Load_LGB(device=device, wandb=wandb_logging)



            


            lgb_train = LGB.convert_data(X_train, Y_train ) # type: ignore
            lgb_val = LGB.convert_data(X_val, Y_val)  # type: ignore
            
            # model = lgb.train(LGB.params, lgb_train, num_boost_round=epochs, valid_sets=[lgb_train, lgb_val], callbacks=[lgb.early_stopping(200), lgb.log_evaluation(10)])
            # model = LGB.train(lgb_train, lgb_val, epochs) # type: ignore

            model = LGB.load_model(evaluation_model)

            loss, accuracy, val_pred = LGB.predict(X_val, Y_val)

                       

            print(f"Final validation set performance:\n\tloss {loss}\n\taccuracy {accuracy}")

            if wandb_logging:
                wandb.log({"test_acc": accuracy, "test_loss": loss})
                wandb.finish()







        else:

            [train_loaders, val_loaders, test_loader, _], num_channels, num_classes = load_partitioned_datasets(num_clients=1, dataset_name=dataset_name)
            
            val_loader = val_loaders[0]   
            train_loader = train_loaders[0]

            test_loader_size = len(test_loader.dataset)


            train_loader = get_dataloaders_subset(train_loader, test_loader_size)

            
            
            # subset_train_loader = []
            # for batch in train_loader:
            #     subset_train_loader.append(batch)        
            #     if len(subset_train_loader) == test_loader_size:
            #         break
            
            # train_loader = subset_train_loader#DataLoader(list(islice(train_loader, len(test_loader))))  # type: ignore


            # print(f"Training on {model_name} with {dataset_name} in {device} using PyTorch {torch.__version__} and Flower {fl.__version__}")
            model = load_model_defination(model_name, num_channels, num_classes, differential_privacy).to(device) # type: ignore
            optimizer = torch.optim.Adam(model.parameters())


            model, optimizer, train_loader = make_private(differential_privacy, model, optimizer, train_loader)


            load_saved_weights(model, filename =evaluation_model)

            

            comment = 'Test_Centralized_('+evaluation_model+')_'+model_name+'_'+dataset_name
            if wandb_logging:
                wandb_init(comment=comment, model_name=model_name, dataset_name=dataset_name)
                wandb.watch(model, log_freq=100)
                
            trn_loss, trn_accuracy, predA = test(model, train_loader)
            val_loss, val_accuracy, _ = test(model, val_loader)
            tst_loss, tst_accuracy, predB = test(model, test_loader)


            
            # pdb.set_trace()




            print(f"Final training set performance:\n\tloss {trn_loss}\n\taccuracy {trn_accuracy}")








            if wandb_logging:
                wandb.log({"train_acc": trn_accuracy, "train_loss": trn_loss})
                wandb.log({"acc": val_accuracy, "loss": val_loss}, step = 100)
                wandb.log({"test_acc": tst_accuracy, "test_loss": tst_loss})
                wandb.finish()
            print(f"Final validation set performance:\n\tloss {val_loss}\n\taccuracy {val_accuracy}")
            print(f"Final test set performance:\n\tloss {tst_loss}\n\taccuracy {tst_accuracy}")

            plot_histogram(predA, predB)
                
            if wandb_logging:
                wandb.finish()
    except Exception as e:
        traceback.print_exc()
        pdb.set_trace()


def train_centralized(epochs, device, wandb_logging=True, savefilename=None, dataset_name='CIFAR10', model_name = 'basic_CNN', differential_privacy=False):


    if model_name == 'lgb':

        import utils.datasets as d 

        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import train_test_split
        data_splits = d.load_incremental_Microsoft_Malware()

        # train_data = data_splits[0][0].dataset.tensors[0]
        # train_labels = data_splits[0][0].dataset.tensors[1]
        # val_data = data_splits[0][1].tensors[0]
        # val_labels = data_splits[0][1].tensors[1]


        # Assuming data_splits[0] contains the train subset
        
        
        try:

            train_subset = data_splits[0][0]
            X = train_subset.dataset.tensors[0].numpy()  # Assuming features are tensor[0]
            Y = train_subset.dataset.tensors[1].numpy()  # Assuming labels are tensor[1]

            # Splitting the train subset into train and validation sets
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)


            comment = model_name+'_Centralized_'+dataset_name
            if not savefilename:
                savefilename = comment 

            if wandb_logging:
                wandb_init(comment=comment, model_name=model_name, dataset_name=dataset_name)
            
            device_name = 'gpu' if str(device) == 'cuda' else str(device)
            LGB = Load_LGB(device=device_name, wandb=wandb_logging)


            lgb_train = LGB.convert_data(X_train, Y_train ) # type: ignore
            lgb_val = LGB.convert_data(X_val, Y_val)  # type: ignore
            
            try:
                LGB.load_model(savefilename)
            except:
                print(f"No saved model found. Training new model")

            model = LGB.train(lgb_train, lgb_val, epochs) # type: ignore

            loss, accuracy, test_pred = LGB.predict(X_test, Y_test)

                       

            print(f"Final validation set performance:\n\tloss {loss}\n\taccuracy {accuracy}")

            if wandb_logging:
                wandb.log({"test_acc": accuracy, "test_loss": loss})
                wandb.finish()

            


              

            LGB.save_model(savefilename)

        except Exception as e:
            traceback.print_exc()
            pdb.set_trace()





        

    else:

        [train_loaders, val_loaders, test_loader, _ ], num_channels, num_classes = load_partitioned_datasets(num_clients=1, dataset_name=dataset_name) 

        # print(f"Training on {model_name} with {dataset_name} in {device} using PyTorch {torch.__version__} and Flower {fl.__version__}")
        model = load_model_defination(model_name, num_channels, num_classes, differential_privacy).to(device) 

        optimizer = torch.optim.Adam(model.parameters())

        print_info(device, model_name, dataset_name)    

        

        train_loader = train_loaders[0]
        val_loader = val_loaders[0]   


        
        if differential_privacy:
            print('Enabling Differential Privacy')
            comment = 'Centralized_dp_'+model_name+'_'+dataset_name
        else:
            comment = 'Centralized_'+model_name+'_'+dataset_name
        

        model, optimizer, train_loader = make_private(differential_privacy, model, optimizer, train_loader)
            
        

        if wandb_logging:
            wandb_init(comment=comment, model_name=model_name, dataset_name=dataset_name)
            wandb.watch(model, log_freq=100)
            

        model, optimizer, val_loss, val_accuracy, _  = train(model, train_loader, val_loader, epochs, optimizer, verbose=False, wandb_logging=wandb_logging)
        loss, accuracy, _ = test(model, test_loader)

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
    parser.add_argument('-dp', '--differential_privacy', action='store_true', help='Enable differential privacy')
    args = parser.parse_args()

    device = get_device()
    if args.evaluation_model:
        evaluate(args.evaluation_model, device, args.wandb_logging, args.dataset_name, args.model_name, args.differential_privacy)
    else:
        for _ in range(args.num_experiments):
            train_centralized(args.num_epochs, device, args.wandb_logging, args.save_filename, args.dataset_name, args.model_name, args.differential_privacy)
        



if __name__ == "__main__":       
    main()
    