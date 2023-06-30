from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import flwr as fl
import wandb
import torch
from utils.client_utils import load_datasets, client_fn
from utils.training_utils import wandb_init, get_device, get_parameters, set_parameters, test
from utils.models import basicCNN as Net
import pdb



# The `evaluate` function will be by Flower called after every round
def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net = Net().to(DEVICE)
    valloader = valloader_all   
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, valloader)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    wandb.log({"acc": accuracy, "loss": loss})
    return loss, {"accuracy": accuracy}

def get_info(num_clients, device, net):
    client_resources = None
    if device.type == "cuda":
        client_resources = {"num_gpus": 1}
    model_name = net.__class__.__name__
    comment = 'Ferderated_'+str(num_clients)+'_'+model_name

    return client_resources, model_name, comment


def run_for_n_rounds(num_rounds, num_clients, net, trainloaders, valloaders):

    client_resources, model_name, comment =  get_info(num_clients,DEVICE, net)
    wandb_init(comment, model_name)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.3,
        min_fit_clients= min(3,num_clients),
        min_evaluate_clients=min(3,num_clients),
        min_available_clients=num_clients,
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net())),
        evaluate_fn=evaluate,  # Pass the evaluation function
    )
    

    fl.simulation.start_simulation(
        client_fn= lambda cid: client_fn(cid, net, trainloaders, valloaders),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),  # Just three rounds
        strategy=strategy,
        client_resources=client_resources,
    )

    wandb.finish()


if __name__ == "__main__":       

    DEVICE = get_device()
    print(
        f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
    )



    num_clientss = 5
    trainloaders, valloaders, testloader , valloader_all = load_datasets(num_clients)
    net = Net().to(DEVICE)
    for i in range(1):
        run_for_n_rounds(5, num_clients, net, trainloaders, valloaders)