from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
import flwr as fl 
from utils.training_utils import test, set_parameters
from utils.models import load_model_defination
import numpy as np
import traceback,pdb

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

class AggregatePrivacyPreservingMetricStrategy(fl.server.strategy.FedAvg):
    
    
    def __init__(self, model, valloader, device,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.valloader = valloader
        self.device = device


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures) 

        print("\naggregating __\n\n")

        
        try:
            # model = load_model_defination(model_name = "efficientnet", num_channels=3, num_classes=10)
            model = self.model
        
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]

            #confidence = [0] * len(weights_results)
            max_agreed_confidence = 0.0
            best_client_index = -1
            client_prediction = []
            for weights , num_examples in weights_results:
                set_parameters(model, weights)
                loss, accuracy, prediction = test(model, self.valloader, self.device)
                #print(f"number of examples {num_examples}")

                client_prediction.append(prediction[0])
                print(f'\n{prediction[0]}\n')
            
            for i,client_prediction in enumerate(client_prediction):
                client_max_confidence = []

                for sample_prediction in client_prediction:
                    max_confidence = max(sample_prediction)
                    client_max_confidence.append(max_confidence)
                
                client_mac = np.mean(client_max_confidence)

                if client_mac > max_agreed_confidence:
                    max_agreed_confidence = client_mac
                    best_client_index = i
            
            print(f"best client index: {best_client_index}")
            print(f"Maximun Aggreed Confidence (MAC) score: {max_agreed_confidence}")

                #pred = prediction [0]
                #m = 0
                #print(f"num examples {num_examples}")
                #for p in pred:
                #    m += max(p)

                #confidence [num_examples] = m
            #n = np.argmax (confidence)
                

            #set_parameters(model, parameters_to_ndarrays(parameters_aggregated))
            #loss, accuracy = test(model, self.valloader, self.device)

            #for weights, num_examples in weights_results:

            #set_parameters(self.model, ndarrays_to_parameters(parameters_aggregated))

            #set_parameters(model, parameters_aggregated)
            #loss, accuracy = test(self.model_temp, self.valloader, self.device)
            
            #    all_losses.append(loss)
            #    all_accuracies.append(accuracy)

            #parameters_aggregated = weight_results [n]
        except Exception as e:
            traceback.print_exc()
            pbdb.set_trace()

        return parameters_aggregated, metrics_aggregated