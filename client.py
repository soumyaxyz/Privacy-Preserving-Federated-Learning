import flwr as fl
import argparse
from utils.client_utils import FlowerClient, load_partitioned_datasets, print_info, get_device
from utils.models import load_model
from utils.client_utils import client_fn
import pdb, traceback




def main():
    parser = argparse.ArgumentParser(description='A description of your program')
    parser.add_argument('-a', '--server_address', type=str, default="[::]", help='Server address')
    parser.add_argument('-p', '--server_port', type=str, default="8080", help='Server port')
    parser.add_argument('-N', '--number_of_total_clients', type=int, default=2, help='Total number of clients')      
    parser.add_argument('-m', '--model_name', type=str, default = "basicCNN", help='Model name') 
    parser.add_argument('-d', '--dataset_name', type=str, default='CIFAR10', help='Dataset name') 
    parser.add_argument('-n', '--client_number', type=int, help='Client number')    
    parser.add_argument('-db','--debug', action='store_false', help='Enable debug mode')
    args = parser.parse_args()
    model = load_model(args.model_name, num_channels=3, num_classes=10)
    model.to(get_device())
    trainloaders, valloaders, _, _ = load_partitioned_datasets(args.number_of_total_clients, dataset_name=args.dataset_name)
    client_defination = client_fn(args.client_number, model, trainloaders, valloaders)
    try:
        fl.client.start_numpy_client(server_address=args.server_address+':'+ args.server_port, client=client_defination)
    except KeyboardInterrupt:
        print("Stopped with by user. Exiting.")
    except Exception as e:
        if args.debug:
            traceback.print_exc()
            pdb.set_trace()
        else:
            print("Stopped with errors. Exiting.")
    



if __name__ == '__main__':
    main()