from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import flwr as fl
import wandb
import torch
from utils.client_utils import load_datasets#, Client_function_wrapper_class
from utils.training_utils import get_parameters
from utils.models import basicCNN as Net
import utils.client_utils as client_utils


DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)


def wandb_init(comment= '', lr ='', optimizer = '', model_name="CNN_1", dataset_name="CIFAR_10"):
    wandb.login()
    wandb.init(
      project="Ferderated-CIFAR_10", entity="soumyabanerjee",
      config={
        "learning_rate": lr,
        "optimiser": optimizer,
        "comment" : comment,
        "model": model_name,
        "dataset": dataset_name,
      }
    )




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
    # wandb.log({"acc": accuracy, "loss": loss})
    return loss, {"accuracy": accuracy}


client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}


def run_for_epoch(num_rounds = 50, num_client=10):
    # wandb_init(comment= 'Ferderated_'+str(num_client))
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.3,
        min_fit_clients= min(3,num_client),
        min_evaluate_clients=min(3,num_client),
        min_available_clients=num_client,
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net())),
        evaluate_fn=evaluate,  # Pass the evaluation function
    )
    c = client_utils.Client_function_wrapper_class(trainloaders, valloaders)

    fl.simulation.start_simulation(
        client_fn=c.client_fn(),
        num_clients=num_client,
        config=fl.server.ServerConfig(num_rounds=num_rounds),  # Just three rounds
        strategy=strategy,
        client_resources=client_resources,
    )

    # wandb.finish()

NUM_CLIENTS = 5
trainloaders, valloaders, testloader , valloader_all = load_datasets(NUM_CLIENTS)
for i in range(3):
    run_for_epoch(5,NUM_CLIENTS)