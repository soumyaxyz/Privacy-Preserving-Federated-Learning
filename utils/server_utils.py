import flwr as fl
from flwr.common import Metrics
import wandb
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from utils.custom_strategy import AggregatePrivacyPreservingMetricStrategy
from utils.training_utils import test, set_parameters, get_parameters
import pdb
#from utils.fedcustom import FedCusTest



class Server_details:
    def __init__(self, model, valloader, wandb_logging, num_clients, device, epochs_per_round =1, mode = 'correct_confident'):
        self.model = model
        self.valloader = valloader
        self.device = device
        self.wandb_logging = wandb_logging
        self.model.to(self.device)
        self.num_clients = num_clients
        self.loss_min =  10000 # inf        
        self.aggregration_mode = self.mode_to_integer(mode)
        self.strategy = self.get_strategy()
        self.epochs_per_round = epochs_per_round
    

    def mode_to_integer(self, mode_text):
        mode_text = mode_text.lower()
        if mode_text == "fedavg":
            return 0
        elif mode_text == "first":
            return 1
        elif mode_text == "first_correct":
            return 2
        elif mode_text == "confident":
            return 3
        elif mode_text == "correct_confident":
            return 4
        else:
            return 0 # default

    def get_certificates(self):
        try:
            certificates=(
                Path(".cache/certificates/ca.crt").read_bytes(),
                Path(".cache/certificates/server.pem").read_bytes(),
                Path(".cache/certificates/server.key").read_bytes()
            )
        except FileNotFoundError:
            print("Certificates not found. Falling back to unsecure mode")
            certificates = None
        return certificates
        

    def get_strategy(self):
        strategy =  AggregatePrivacyPreservingMetricStrategy(  # fl.server.strategy.FedAvg
                    mode=self.aggregration_mode,
                    model=self.model,
                    valloader=self.valloader,
                    device=self.device,
                    fraction_fit=0.3,
                    fraction_evaluate=0.3,
                    # min_fit_clients= min(2,self.num_clients),
                    # min_evaluate_clients=min(2,self.num_clients),
                    # min_available_clients=self.num_clients,
                    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(self.model)),
                    on_fit_config_fn=lambda server_round :  self.fit_config(server_round), # type: ignore
                    on_evaluate_config_fn=self.evaluate_config, # type: ignore
                    evaluate_fn=lambda server_round, parameters, config : self.get_evaluate_fn(
                                                                                            server_round, parameters, config, 
                                                                                            self.model, self.valloader, self.device, 
                                                                                            self.wandb_logging
                                                                                          ),
                    evaluate_metrics_aggregation_fn=self.weighted_average
                )
        return strategy
    
    

    def fit_config(self, server_round: int):
        """Return training configuration dict for each round.
        """
        config = {
            "batch_size": 32,
            "local_epochs": self.epochs_per_round ,
        }
        return config

    def evaluate_config(self, server_round: int):
        """Return evaluation configuration dict for each round.       
        """
        print(f"[SERVER round {server_round}], Evaluate config \n\n")
        config = {
            "server_round": server_round,
            "local_epochs": self.epochs_per_round ,
        }
        return config


    def weighted_average(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"accuracy": sum(accuracies) / sum(examples)} # type: ignore


    def get_evaluate_fn(self, server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar],
            model, valloader, device, wandb_logging
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            
            # pdb.set_trace()
            
            print(f"[SERVER round {server_round}], get_evaluate_fn \n\n")


            set_parameters(model, parameters)  # Update model with the latest parameters
            print(f"Server-side evaluation started\n\n\n")
            loss, accuracy, _ = test(model, valloader, device)
            print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
            if wandb_logging:
                wandb.log({"acc": accuracy, "loss": loss}, step=server_round*self.epochs_per_round)            
            
            return loss, {"accuracy": accuracy}