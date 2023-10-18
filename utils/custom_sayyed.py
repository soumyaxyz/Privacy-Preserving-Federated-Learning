from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
import flwr as fl 
from utils.training_utils import test, set_parameters
from utils.models import load_model_defination

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    
    
    def __init__(self, valloader, device,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model = load_model_defination(model_name = "efficientnet", num_channels=3, num_classes=10) 
        self.valloader = valloader
        self.device = device


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures) 

        weights_results = [
        (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        for _, fit_res in results
        ]

        all_loss = []
        all_accuracy = []
        for weights, num_examples in weights_results:
            set_parameters(self.model, ndarrays_to_parameters(weights))
            loss, accuracy = test(self.model, self.valloader, self.device)
            all_loss.append(loss)
            all_accuracy.append(accuracy)

        #for weights, num_examples in weights_results:

        #set_parameters(self.model, ndarrays_to_parameters(parameters_aggregated))

        #set_parameters(self.model_temp, parameters_aggregated)
        #loss, accuracy = test(self.model_temp, self.valloader, self.device)
        
        #    all_losses.append(loss)
        #    all_accuracies.append(accuracy)

        print("done")
        return parameters_aggregated, metrics_aggregated