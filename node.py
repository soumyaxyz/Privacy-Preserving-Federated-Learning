import flwr as fl
from flwr.common.logger import log
from logging import WARNING, INFO
import socket
import threading
import argparse, wandb
from utils.datasets import load_partitioned_datasets, merge_dataloaders
from utils.training_utils import print_info, save_model, wandb_init, get_device,  test
from utils.client_utils import client_fn, get_certificate
from utils.models import load_model_defination
from utils.server_utils import Server_details
import pdb, traceback



class Node:
    def __init__(self, args):
        self.server_address_port = args.initial_server_address+':'+ args.initial_server_port
        self.number_of_total_clients = args.number_of_total_clients
        self.debug = args.debug
        self.args = args

        device = get_device()

        [trainloaders, valloaders, _, _], num_channels, num_classes = load_partitioned_datasets(args.number_of_total_clients, dataset_name=args.dataset_name)
        model = load_model_defination(args.model_name, num_channels=num_channels, num_classes=num_classes)

        model.to(device)    
        
        if args.overfit_patience == -1:
            args.overfit_patience = 100000 # Initial patience, allow overfitting

        if args.secure:
            self.certificate = get_certificate()
        else:
            self.certificate = None

        if self.debug:    
            print(f'certificate: {self.certificate}')
            
        
        print_info(device, args.model_name, args.dataset_name)

        self.client_defination = client_fn(args.client_number, model, trainloaders, valloaders, 
                                    args.number_of_total_clients, args.wandb_logging, args.dataset_name, 
                                    args.overfit_patience, simulation=args.headless
                                    )
        
        # self.run_client_service(self.server_address_port)
        # log(INFO, "Starting server")
        self.run_server_service()


        
    def run_client_service(self, server_address_port):
        try:
            fl.client.start_numpy_client(
                                        server_address = server_address_port, 
                                        client = self.client_defination,
                                        root_certificates = self.certificate
                                        )
        except KeyboardInterrupt:
            print("Stopped with by user. Exiting.")
        except Exception as e:
            if self.debug:
                traceback.print_exc()
                pdb.set_trace()
            else:
                print("Stopped with errors. Exiting.")


    def run_server_service(self): 
        loaders, num_channels,num_classes = load_partitioned_datasets(self.args.number_of_total_clients, dataset_name=self.args.dataset_name)
        model = load_model_defination(self.args.model_name, num_channels=num_channels, num_classes=num_classes)    
        [trainloaders,_,test_loader , valloader_all] = loaders

        trainloader_all = merge_dataloaders(trainloaders) 

        device = get_device()
        print_info(device, self.args.model_name, self.args.dataset_name)

        # log(INFO, "defining server")
        server_details = Server_details(model = model, 
                                        trainloader = trainloader_all, 
                                        valloader = valloader_all, 
                                        wandb_logging = self.args.wandb_logging, 
                                        num_clients = self.args.number_of_total_clients, 
                                        device = device, 
                                        epochs_per_round = self.args.epochs_per_round, 
                                        mode = self.args.federated_learning_mode)

        comment = self.args.comment+'_'+str(self.args.number_of_total_clients)+'_'+self.args.federated_learning_mode+'_'+self.args.model_name+'_'+self.args.dataset_name

        if self.args.wandb_logging:        
            wandb_init(comment=comment, model_name=self.args.model_name, dataset_name=self.args.dataset_name)

        if self.args.secure:
            certificates = server_details.get_certificates()
        else:
            certificates = None
        
        try :
            fl.server.start_server(
                                    server_address = self.server_address_port, 
                                    config=fl.server.ServerConfig(num_rounds=self.args.number_of_FL_rounds), 
                                    strategy=server_details.strategy,
                                    certificates=certificates
                                )
            
            
            
            loss, accuracy, _ = test(model, test_loader)
            save_model(model,filename =comment, print_info=True)
        except KeyboardInterrupt:
            print("Stopped with by user. Exiting.")
        except Exception as e:
            if self.debug:
                traceback.print_exc()
                pdb.set_trace()
            else:
                print("Stopped with errors. Exiting.")
        

        if self.args.wandb_logging:
            try:
                wandb.log({"test_acc": accuracy, "test_loss": loss}) # type: ignore
            except UnboundLocalError:
                pass
            wandb.finish()




def main():    
    
    parser = argparse.ArgumentParser(description='A description of your program') 
    parser.add_argument('-a', '--initial_server_address', type=str, default="[::]", help='Server address')
    parser.add_argument('-p', '--initial_server_port', type=str, default="8080", help='Server port')    
    parser.add_argument('-m', '--model_name', type=str, default = "basicCNN", help='Model name')
    parser.add_argument('-fl', '--federated_learning_mode', type=str, default='fedavg', help='How to combine the clients weights:fedavg, first,  confident, correct_confident')
    parser.add_argument('-c', '--comment', type=str, default='Federated_', help='Comment for this run')
    parser.add_argument('-d', '--dataset_name', type=str, default='CIFAR10', help='Dataset name')
    parser.add_argument('-r', '--number_of_FL_rounds', type=int, default = 3, help='Number of rounds of Federated Learning')  
    parser.add_argument('-N', '--number_of_total_clients', type=int, default=2, help='Total number of clients')  
    parser.add_argument('-w', '--wandb_logging', action='store_true', help='Enable wandb logging')
    parser.add_argument('-db','--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-s', '--secure', action='store_true', help='Enable secure mode')
    parser.add_argument('-e', '--epochs_per_round', type=int, default = 1, help='Epochs of training in client per round')
    parser.add_argument('-n', '--client_number', type=int, help='Client number')    
    parser.add_argument('-o', '--overfit_patience', type=int, default=-1, help='Patience after which to stop training, to prevent overfitting')
    parser.add_argument('-hl','--headless', action='store_true', help='Enable headless mode')
    args = parser.parse_args()

    node = Node(args)

    
    

    
    
    



if __name__ == '__main__':
    main()