import flwr as fl
import argparse
from utils.client_utils import load_partitioned_datasets, load_datasets
from utils.models import load_model
from utils.server_utils import post_round_evaluate_function

class Server_configs:
    def __init__(self, model, valloder):
        self.model = model
        self.valloder = valloder

        self.strategy = fl.server.strategy.FedAvg(
                    fraction_fit=0.3,
                    fraction_evaluate=0.3,
                    # min_fit_clients= min(3,self.num_clients),
                    # min_evaluate_clients=min(3,self.num_clients),
                    # min_available_clients=self.num_clients,
                    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net())),
                    evaluate_fn=lambda server_round, parameters, config : post_round_evaluate_function(server_round, parameters, config, self.model, self.valloaders)
                )


def main():
    # parser = argparse.ArgumentParser(description='A description of your program')
    # parser.add_argument('-a', '--server address', type=str, default="[::]", help='Number of experiments')
    # parser.add_argument('-p', '--server port', type=str, default="8080", help='Number of experiments')
    # args = parser.parse_args()
    
    parser = argparse.ArgumentParser(description='A description of your program')     
    parser.add_argument('-m', '--model_name', type=str, default = "basicCNN", help='Model name')
    parser.add_argument('-r', '--number_of_FL_rounds', type=int, default = 3, help='Number of rounds of Federated Learning')  
    parser.add_argument('-N', '--number_of_total_clients', type=int, default=2, help='Total number of clients')  
    args = parser.parse_args()

    model = load_model(args.model_name, num_channels=3, num_classes=10)     
    _,_,_ , valloader_all = load_datasets(args.number_of_total_clients)
    sc = Server_configs(model, valloader_all)

    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=args.number_of_FL_rounds), strategy=sc.strategy)



if __name__ == '__main__':
    main()