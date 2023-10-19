from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
import flwr as fl 
from utils.training_utils import test, set_parameters
from utils.models import load_model_defination
from .aggregate import aggregate, weighted_loss_avg
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
    
    
    def __init__(self, mode, model, valloader, device,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.model = model
        self.valloader = valloader
        self.device = device


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if self.mode == 0:
            return super().aggregate_fit(server_round, results, failures)

        # parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures) 

        print("\naggregating __\n\n")

        # MODE 0 = FEDAVG
        # MODE 1 = RETURN FIRST
        # MODE 2 = RETURN FIRST CORRECT
        # MODE 3 = RETURN MOST CONFIDENT
        # MODE 4 = RETURN CORRECT AND MOST CONFIDENT

        
        try:
            # model = load_model_defination(model_name = "efficientnet", num_channels=3, num_classes=10)
            # model = self.model
        
            # Convert results
            weights_results = [ (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)  for _, fit_res in results ]

            #confidence = [0] * len(weights_results)
            max_agreed_confidence = 0.0
            best_client_index = -1
            client_prediction = []
            for weights , num_examples in weights_results:
                set_parameters(self.model, weights)
                loss, accuracy, prediction = test(self.model, self.valloader, self.device)
                #print(f"number of examples {num_examples}")

                (confidence, eval_results) = prediction # type: ignore

                if self.mode == 1 or self.mode == 4:                    
                    client_prediction.append(np.mean(confidence ))
                    print(f'\n{np.mean(confidence)}\n')
                if self.mode == 2 or self.mode == 3:
                    filtered_confidence = confidence[eval_results == 1]
                    client_prediction.append(np.mean(filtered_confidence ))
                    print(f'\n{np.mean(filtered_confidence)}\n')


            most_confident_model_index  =  np.argmax(client_prediction) 

            # select_client_weights = [weights_results [most_confident_model_index]]

            # weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]  

            _, fit_res = results[most_confident_model_index]

            weights_results_1 = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)]

            



            parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results_1))
            # parameters_aggregated = weights_results


            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No fit_metrics_aggregation_fn provided")
            
            
            print(f"Selected client index: {most_confident_model_index}")
            print(f"Maximun Aggreed Confidence (MAC) score: {max_agreed_confidence}")

            



        except Exception as e:
            traceback.print_exc()
            pdb.set_trace()

        return parameters_aggregated, metrics_aggregated