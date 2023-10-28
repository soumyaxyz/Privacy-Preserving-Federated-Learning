from logging import WARNING, INFO
from typing import Callable, Dict, List, Optional, Tuple, Union
import flwr as fl
from utils.lib import try_catch 
from utils.training_utils import test, set_parameters
from utils.models import load_model_defination
from .aggregate import aggregate, weighted_loss_avg
import numpy as np
import traceback,pdb
import csv

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
        self.save_confidence = True

    @try_catch
    def save_to_file(self, vallues, round, file_name="confidences"):
        csv_file_name = f"./results/{file_name}_{round}.csv"

        # Write the confidences list to the CSV file
        with open(csv_file_name, mode='w', newline='') as csv_file:
            if file_name == "confidences":
                for confidence in vallues:
                    csv_file.write(','.join(map(str, confidence)) + '\n')
            else:
                csv_file.write(','.join(map(str, vallues)) + '\n')
        log(INFO, f"{file_name} saved to {csv_file_name}")


    @try_catch
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if self.mode == 0:
            return super().aggregate_fit(server_round, results, failures)

        # MODE 0 = FEDAVG
        # MODE 1 = RETURN FIRST
        # MODE 2 = RETURN MOST CONFIDENT
        # MODE 3 = RETURN CORRECT AND MOST CONFIDENT

        
        # try:
            
        weights_results = [ (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)  for _, fit_res in results ]

        confidences = []
        eval_results_sum = None

        if self.mode == 1:
            most_confident_model_index = 0
        else:
            client_prediction = []
            
            for weights , num_examples in weights_results:
                set_parameters(self.model, weights)
                _, _, prediction = test(self.model, self.valloader, self.device)

                (confidence, eval_results) = prediction # type: ignore
                if eval_results_sum is None:
                    eval_results_sum = eval_results
                else:
                    eval_results_sum += eval_results

                if self.save_confidence:
                    confidences.append(confidence)

                if self.mode == 2:                    
                    client_prediction.append(np.mean(confidence ))
                if self.mode == 3:
                    filtered_confidence = confidence[eval_results == 1]
                    client_prediction.append(np.mean(filtered_confidence ))

            most_confident_model_index  =  np.argmax(client_prediction) 
            # log(INFO, f'\n\n number of clients {len(weights_results)} ,  Maximum aggeed correct = {max(eval_results_sum)}\n\n') # type: ignore
        if self.save_confidence:
            self.save_to_file(confidences, server_round)
            self.save_to_file(eval_results_sum, server_round, file_name = "eval_results")
        
        _, client_result = results[most_confident_model_index]
        selected_weights_results = [(parameters_to_ndarrays(client_result.parameters), client_result.num_examples)]

        parameters_selected = ndarrays_to_parameters(aggregate(selected_weights_results))

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        
        
        # print(f"Selected client index: {most_confident_model_index}")
        log(INFO, f"Selected client index: {most_confident_model_index}")

                



        # except Exception as e:
        #     traceback.print_exc()
        #     pdb.set_trace()

        return parameters_selected, metrics_aggregated