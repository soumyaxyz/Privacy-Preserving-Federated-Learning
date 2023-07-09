import flwr as fl
import argparse, wandb
from utils.training_utils import print_info, save_model, wandb_init, get_device, get_parameters, set_parameters, test
from utils.client_utils import load_partitioned_datasets
from utils.models import load_model
from utils.server_utils import get_evaluate_fn, fit_config
import pdb, traceback

class Server_configs:
    def __init__(self, model, valloader, wandb_logging, num_clients, device):
        self.model = model
        self.valloader = valloader
        self.device = device
        self.wandb_logging = wandb_logging
        self.model.to(self.device)
        self.num_clients = num_clients

        self.strategy = fl.server.strategy.FedAvg(
                    fraction_fit=0.3,
                    fraction_evaluate=0.3,
                    # min_fit_clients= min(2,self.num_clients),
                    # min_evaluate_clients=min(2,self.num_clients),
                    # min_available_clients=self.num_clients,
                    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(self.model)),
                    evaluate_fn=lambda server_round, parameters, config : get_evaluate_fn(server_round, parameters, config, self.model, self.valloader, self.device, self.wandb_logging)
                )



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
    parser.add_argument('-c', '--comment', type=str, default='Federated_', help='Comment for this run')
    parser.add_argument('-d', '--dataset_name', type=str, default='CIFAR10', help='Dataset name')
    parser.add_argument('-r', '--number_of_FL_rounds', type=int, default = 3, help='Number of rounds of Federated Learning')  
    parser.add_argument('-N', '--number_of_total_clients', type=int, default=2, help='Total number of clients')  
    parser.add_argument('-w', '--wandb_logging', action='store_true', help='Enable wandb logging')
    parser.add_argument('-db','--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    

    model = load_model(args.model_name, num_channels=3, num_classes=10)     
    _,_,test_loader , valloader_all = load_partitioned_datasets(args.number_of_total_clients, dataset_name=args.dataset_name)

    device = get_device()
    print_info(device, args.model_name, args.dataset_name)


    sc = Server_configs(model, valloader_all, args.wandb_logging, args.number_of_total_clients, device)

    if args.wandb_logging:
        comment = args.comment+str(args.number_of_total_clients)+'_'+args.model_name+'_'+args.dataset_name
        wandb_init(comment=comment, model_name=args.model_name, dataset_name=args.dataset_name)

    try :
        fl.server.start_server(server_address = args.server_address+':'+ args.server_port, config=fl.server.ServerConfig(num_rounds=args.number_of_FL_rounds), strategy=sc.strategy)
    except KeyboardInterrupt:
        print("Stopped with by user. Exiting.")
    except Exception as e:
        if args.debug:
            traceback.print_exc()
            pdb.set_trace()
        else:
            print("Stopped with errors. Exiting.")
    try:
        loss, accuracy = test(model, test_loader)
        save_model(model,filename =comment, print_info=True)
    except Exception as e:
        traceback.print_exc()
        pdb.set_trace()

    if args.wandb_logging:
        wandb.log({"test_acc": accuracy, "test_loss": loss})
        wandb.finish()


if __name__ == '__main__':
    main()