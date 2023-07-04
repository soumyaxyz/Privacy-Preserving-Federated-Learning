from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import flwr as fl
import wandb
import torch
from utils.client_utils import load_datasets, client_fn
from utils.training_utils import print_info, save_model, wandb_init, get_device, get_parameters, set_parameters, test
from utils.models import basicCNN as Net
from utils.server_utils import post_round_evaluate_function
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

    def __init__(self, num_clients, wandb_logging = True):
        super(Simulator, self).__init__()
        self.wandb_logging  = wandb_logging
        self.device         = get_device()
        self.num_clients    = num_clients
        self.trainloaders, self.valloaders, self.testloader , self.valloader_all = load_datasets(self.num_clients)
        if self.device.type == "cuda":
            self.client_resources = {"num_gpus": 1}
        else:
            self.client_resources = None
        self.net = Net().to(self.device)
        

    # The `evaluate` function will be by Flower called after every round
    

    def get_info(self):        
        model_name = self.net.__class__.__name__
        comment = 'Ferderated_'+str(self.num_clients)+'_'+model_name
        return  model_name, comment
   

    def run_for_n_rounds(self, num_rounds):     
        """
        Runs the federated learning for a specified number of rounds.

        Args:
            num_rounds (int): The number of rounds to run the federated learning.

        Returns:
            None
        """
        if self.wandb_logging:
            model_name, comment =  self.get_info()
            wandb_init(comment, model_name)

        strategy = fl.server.strategy.FedAvg(
            fraction_fit=0.3,
            fraction_evaluate=0.3,
            min_fit_clients= min(3,self.num_clients),
            min_evaluate_clients=min(3,self.num_clients),
            min_available_clients=self.num_clients,
            initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net())),
            # evaluate_fn=post_round_evaluate_function,  # Pass the evaluation function
            evaluate_fn=lambda server_round, parameters, config : post_round_evaluate_function(server_round, parameters, config, self.net, self.valloaders_all),  # Pass the evaluation function
        )
        

        fl.simulation.start_simulation(
            client_fn= lambda cid: client_fn(cid, self.net, self.trainloaders, self.valloaders),
            num_clients=self.num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),  # Just three rounds
            strategy=strategy,
            client_resources=self.client_resources,
        )

        if self.wandb_logging:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='A description of your program')

    # Add arguments here
    parser.add_argument('--num_clients', type=int, default=5, help='Number of clients')
    parser.add_argument('--num_experiments', type=int, default=1, help='Number of experiments')
    parser.add_argument('--num_rounds', type=int, default=5, help='Number of rounds')
    parser.add_argument('--save_location', type=str, default='./saved_models/', help='Save location')
    parser.add_argument('--wandb_logging', action='store_true', help='Enable wandb logging')

    args = parser.parse_args()

    
    simulator = Simulator(args.num_clients, args.wandb_logging)
    print_info(simulator.device)    
    for _ in range(args.num_experiments):
        simulator.run_for_n_rounds(args.num_rounds)

    net = Net().to(simulator.device)  
    simulator.set_parameters(net, parameters)
    _, comment = simulator.get_info()
    save_path = args.save_location + comment + ".pt"
    save_model(net, optim, save_path)





if __name__ == "__main__":
    main()
         
    