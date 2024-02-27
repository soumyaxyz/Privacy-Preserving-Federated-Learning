import flwr as fl
import argparse, wandb
from utils.training_utils import print_info, save_model, wandb_init, get_device,  test
from utils.datasets import ContinuousDatasetWraper, load_partitioned_continous_datasets, load_partitioned_datasets, merge_dataloaders
from utils.models import load_model_defination
from utils.server_utils import Server_details
import pdb, traceback


def run_server_once(args, model, loaders):
    [trainloaders,_,test_loader , valloader_all] = loaders

    trainloader_all = merge_dataloaders(trainloaders) 

    device = get_device()
    print_info(device, args.model_name, args.dataset_name)


    server_details = Server_details(model = model, 
                                    trainloader = trainloader_all, 
                                    valloader = valloader_all, 
                                    wandb_logging = args.wandb_logging, 
                                    num_clients = args.number_of_total_clients, 
                                    device = device, 
                                    epochs_per_round = args.epochs_per_round, 
                                    mode = args.federated_learning_mode)

    comment = args.comment+'_'+str(args.number_of_total_clients)+'_'+args.federated_learning_mode+'_'+args.model_name+'_'+args.dataset_name

    if args.wandb_logging:        
        wandb_init(comment=comment, model_name=args.model_name, dataset_name=args.dataset_name)

    if args.secure:
        certificates = server_details.get_certificates()
    else:
        certificates = None

    try :
        fl.server.start_server(
                                server_address = args.server_address+':'+ args.server_port, 
                                config=fl.server.ServerConfig(num_rounds=args.number_of_FL_rounds), 
                                strategy=server_details.strategy,
                                certificates=certificates
                            )
        
        loss, accuracy, _ = test(model, test_loader)
        save_model(model,filename =comment, print_info=True)
    except KeyboardInterrupt:
        print("Stopped with by user. Exiting.")
    except Exception as e:
        if args.debug:
            traceback.print_exc()
            pdb.set_trace()
        else:
            print("Stopped with errors. Exiting.")
    

    if args.wandb_logging:
        try:
            wandb.log({"test_acc": accuracy, "test_loss": loss}) # type: ignore
        except UnboundLocalError:
            pass
        wandb.finish()

    return comment





def main():
    """
    Main function that executes the program.

    Parses command line arguments, loads model and datasets,
    initializes server configurations, starts the Federated Learning server,
    handles exceptions, tests the model, and logs test accuracy and loss.

    Args:
        -a, --server_address (str): The server address. Default is "[::]".
        -p, --server_port (str): The server port. Default is "8080".
        -m, --model_name (str): The model name. Default is "basicCNN".
        -c, --comment (str): The comment for this run. Default is "Federated_".
        -d, --dataset_name (str): The dataset name. Default is "CIFAR10".
        -r, --number_of_FL_rounds (int): The number of rounds of Federated Learning. Default is 3.
        -N, --number_of_total_clients (int): The total number of clients. Default is 2.
        -w, --wandb_logging: Enable wandb logging.
        -db, --debug: Enable debug mode.

    Returns:
        None
    """
    
    
    parser = argparse.ArgumentParser(description='A description of your program') 
    parser.add_argument('-a', '--server_address', type=str, default="[::]", help='Server address')
    parser.add_argument('-p', '--server_port', type=str, default="8080", help='Server port')    
    parser.add_argument('-m', '--model_name', type=str, default = "basicCNN", help='Model name')
    parser.add_argument('-fl', '--federated_learning_mode', type=str, default='confident', help='How to combine the clients weights:fedavg, first,  confident, correct_confident')
    parser.add_argument('-c', '--comment', type=str, default='Federated_', help='Comment for this run')
    parser.add_argument('-d', '--dataset_name', type=str, default='CIFAR10', help='Dataset name')
    parser.add_argument('-r', '--number_of_FL_rounds', type=int, default = 3, help='Number of rounds of Federated Learning')  
    parser.add_argument('-N', '--number_of_total_clients', type=int, default=2, help='Total number of clients')  
    parser.add_argument('-w', '--wandb_logging', action='store_true', help='Enable wandb logging')
    parser.add_argument('-db','--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-s', '--secure', action='store_true', help='Enable secure mode')
    parser.add_argument('-e', '--epochs_per_round', type=int, default = 1, help='Epochs of training in client per round')
    args = parser.parse_args()
    
    # 
    

    if 'continuous' in args.dataset_name: 
        continous_datasets = ContinuousDatasetWraper(args.dataset_name)
        saved_model_names = []

        _, _, num_channels, num_classes = continous_datasets.splits[0]
        model = load_model_defination(args.model_name, num_channels, num_classes)

        for i,dataset_split in enumerate(continous_datasets.splits):
            loaders, num_channels_i, num_classes_i = load_partitioned_continous_datasets(num_clients=args.number_of_total_clients, dataset_split=dataset_split) 
            assert num_channels_i == num_channels
            assert num_classes_i == num_classes
            
            comments = run_server_once(args, model, loaders) 
            saved_model_names.append(comments)
            
        
    else:    
        
        loaders, num_channels, num_classes = load_partitioned_datasets(args.number_of_total_clients, dataset_name=args.dataset_name)
        model = load_model_defination(args.model_name, num_channels, num_classes) 
        comments = run_server_once(args, model, loaders)
        
        
        
        





    


if __name__ == '__main__':
    main()