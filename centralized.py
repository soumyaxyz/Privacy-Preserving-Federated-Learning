import wandb
import torch
import argparse
import pdb,traceback
import numpy as np
from utils.datasets import load_dataset, load_partitioned_dataloaders, get_dataloaders_subset
from utils.plot_utils import plot_histogram
from utils.training_utils import make_private, save_model, wandb_init,  print_info, get_device, train, test, load_model as load_saved_weights
from utils.models import Load_LGB, extract_model_names, load_model_defination, load_non_pytorch_model_defination 






def evaluate(evaluation_model, device, wandb_logging=True,  dataset_name='CIFAR10', model_name = 'efficientnet', differential_privacy=False):
    


    print_info(device, model_name, dataset_name, eval=True)    
    try:

        if  model_name == 'lgb':
            
            
            dataset = load_dataset(dataset_name)
            _, _, _, _, X_test, y_test =   dataset.get_X_y()         
            

            pdb.set_trace()
            

            comment = model_name+'_Centralized_'+dataset_name

            if wandb_logging:
                wandb_init(comment=comment, model_name=model_name, dataset_name=dataset_name)

            param_id = evaluation_model[-1]

            LGB = Load_LGB(device=device, param_id= param_id, wandb=wandb_logging)

            
            model = LGB.load_model(evaluation_model)

            loss, accuracy, val_pred = LGB.predict(X_test, y_test)

                       

            print(f"Final validation set performance:\n\tloss {loss}\n\taccuracy {accuracy}")

            if wandb_logging:
                wandb.log({"test_acc": accuracy, "test_loss": loss})
                wandb.finish()

        else:

            [train_loaders, val_loaders, test_loader, _], num_channels, num_classes = load_partitioned_dataloaders(num_clients=1, dataset_name=dataset_name)
            
            val_loader = val_loaders[0]   
            train_loader = train_loaders[0]

            test_loader_size = len(test_loader.dataset)


            train_loader = get_dataloaders_subset(train_loader, test_loader_size)
            
           
            model = load_model_defination(model_name, num_channels, num_classes, differential_privacy).to(device) 
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
    

    if model_name in extract_model_names(load_non_pytorch_model_defination):

        
        try:

            
            dataset = load_dataset(dataset_name)
            X_train, y_train, X_val, y_val, X_test, y_test =   dataset.get_X_y()

            
            num_channels = X_train.shape[1]
            num_classes = len(np.unique(y_train))





           
            comment = model_name+'_Centralized_'+dataset_name
            if not savefilename:
                savefilename = comment 

            if wandb_logging:
                wandb_init(comment=comment, model_name=model_name, dataset_name=dataset_name)
            
            
            # model_def = Load_LGB(device=device, wandb=wandb_logging)

            model_def = load_non_pytorch_model_defination(model_name=model_name, device=device, num_channels=num_channels, num_classes=num_classes, wandb=wandb_logging)


            lgb_train = model_def.convert_data(X_train, y_train )
            lgb_val = model_def.convert_data(X_val, y_val)  
            
            try:
                model_def.load_model(savefilename)
            except:
                print(f"No saved model found. Training new model")

            model = model_def.train(lgb_train, lgb_val, epochs) 

            loss, accuracy, test_pred = model_def.predict(X_test, y_test)

                       

            print(f"Final validation set performance:\n\tloss {loss}\n\taccuracy {accuracy}")

            if wandb_logging:
                wandb.log({"test_acc": accuracy, "test_loss": loss})
                wandb.finish()

            


              

            model_def.save_model(savefilename)

        except Exception as e:
            traceback.print_exc()
            pdb.set_trace()





        

    else:

        [train_loaders, val_loaders, test_loader, _ ], num_channels, num_classes = load_partitioned_dataloaders(num_clients=1, dataset_name=dataset_name) 

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
    