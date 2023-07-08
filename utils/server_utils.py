import flwr as fl
from collections import OrderedDict
import wandb
from typing import Dict, List, Optional, Tuple
from utils.training_utils import test, set_parameters
from utils.client_utils import get_training_epoch


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 16,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def get_evaluate_fn(server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
        model, valloader, device, wandb_logging
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:          
        set_parameters(model, parameters)  # Update model with the latest parameters
        loss, accuracy = test(model, valloader, device)
        print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
        if wandb_logging:
            wandb.log({"acc": accuracy, "loss": loss}, step=server_round*get_training_epoch())
        return loss, {"accuracy": accuracy}