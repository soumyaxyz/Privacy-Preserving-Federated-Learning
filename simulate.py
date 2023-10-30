from typing import Dict, List 
import flwr as fl
import wandb
import torch
from utils.datasets import load_partitioned_datasets, merge_dataloaders
from utils.models import load_model_defination
from utils.client_utils import client_fn
from utils.server_utils import Server_details
from utils.training_utils import print_info, save_model, wandb_init, get_device, get_parameters, set_parameters, test
from utils.models import basicCNN as Net
import pdb
import argparse

class Simulator(object):
    """
    A class representing a simulator for privacy-preserving federated learning.

    Attributes:
        - num_clients: An integer representing the number of clients in the simulation.
        - logging: A boolean indicating whether logging is enabled.
        - wandb_logging: A boolean indicating whether logging using wandb is enabled.
        - device: A torch.device object representing the device on which the simulation is run.
        - trainloaders: A list of training data loaders for each client.
        - valloaders: A list of validation data loaders for each client.
        - testloader: A data loader for testing the trained model.
        - valloader_all: A data loader for validation using all client data.
        - client_resources: A dictionary representing the resources available to each client.
        - net: An instance of the Net class representing the neural network model.

    Methods:
        - evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            Performs evaluation of the model on the server side.

        - get_info(num_clients, device, net) -> Tuple[Dict[str, int], str, str]:
            Returns information about the simulator.

        - run_for_n_rounds(num_rounds):
            Runs the simulation for a specified number of rounds.
    """

    def __init__(self, args, comment = "Simulation"):
        super(Simulator, self).__init__()
        self.wandb_logging  = args.wandb_logging
        self.device         = get_device()
        self.num_clients    = args.num_clients
        self.model_name     = args.model_name
        self.dataset_name   = args.dataset_name
        self.comment        = comment
        self.epochs_per_round = args.epochs_per_round
        self.federated_learning_mode = args.federated_learning_mode
        [self.trainloaders, self.valloaders, self.testloader , self.valloader_all], self.num_channel , self.num_classes = load_partitioned_datasets(self.num_clients, dataset_name=self.dataset_name) # type: ignore
        self.trainloader_all = merge_dataloaders(self.trainloaders) 
        if self.device.type == "cuda":
            self.client_resources = { "num_gpus": 1, "num_cpus": 1}            
        else:
            # self.client_resources = None
            self.client_resources = { "num_cpus": 1}
        
        self.net = load_model_defination(self.model_name).to(self.device)
        

    # The `evaluate` function will be by Flower called after every round
    

    

    def run_for_n_rounds(self, num_rounds):     
        """
        Runs the federated learning for a specified number of rounds.

        Args:
            num_rounds (int): The number of rounds to run the federated learning.

        Returns:
            None
        """
        
        print_info(self.device)    

        if self.wandb_logging:
            wandb_init(comment=self.comment, model_name=self.model_name, dataset_name=self.dataset_name)

        # server_details = Server_details(self.net, self.valloader_all, self.wandb_logging, self.num_clients, self.device, num_rounds)
        server_details = Server_details(model = self.net, 
                                    trainloader = self.trainloader_all, 
                                    valloader = self.valloader_all, 
                                    wandb_logging = self.wandb_logging, 
                                    num_clients = self.num_clients, 
                                    device = self.device, 
                                    epochs_per_round = self.epochs_per_round, 
                                    mode = self.federated_learning_mode)

        

        fl.simulation.start_simulation(
            client_fn= lambda cid: client_fn(cid, self.net, self.trainloaders, self.valloaders, N=self.num_clients, wandb_logging=False, dataset_name=self.dataset_name, simulation=True),
            num_clients=self.num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),  
            strategy=server_details.strategy,
            client_resources=self.client_resources,
        )

        if self.wandb_logging:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='A description of your program')

    # Add arguments here
    parser.add_argument('-n', '--num_clients', type=int, default=5, help='Number of clients')
    parser.add_argument('-r', '--num_rounds', type=int, default=5, help='Number of rounds')   
    parser.add_argument('-m', '--model_name', type=str, default = "basicCNN", help='Model name')
    parser.add_argument('-c', '--comment', type=str, default='Simulated_', help='Comment for this run')
    parser.add_argument('-fl', '--federated_learning_mode', type=str, default='correct_confident', help='How to combine the clients weights:fedavg, first,  confident, correct_confident')
    parser.add_argument('-d', '--dataset_name', type=str, default='CIFAR10', help='Dataset name')
    parser.add_argument('-w', '--wandb_logging', action='store_true', help='Enable wandb logging')
    parser.add_argument('-db', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-s', '--secure', action='store_true', help='Enable secure mode')
    parser.add_argument('-e', '--epochs_per_round', type=int, default=1, help='Epochs of training in client per round')
    parser.add_argument('-o', '--overfit_patience', type=int, default=-1, help='Patience after which to stop training, to prevent overfitting')
    parser.add_argument('-hl', '--headless', action='store_true', help='Enable headless mode')
    args = parser.parse_args()

    comment = args.comment+'_'+str(args.num_clients)+'_'+args.model_name+'_'+args.dataset_name

    
    simulator = Simulator(args, comment)
    simulator.run_for_n_rounds(args.num_rounds)

    # net = Net().to(simulator.device)  
    # simulator.set_parameters(net, parameters)

    

    
    
    save_model(simulator.net ,filename = comment, print_info = True)





if __name__ == "__main__":
    # print("Simulator mode deprecated, use server client mode instead.")
    main()
         
    