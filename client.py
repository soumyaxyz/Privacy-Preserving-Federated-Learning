import flwr as fl
import argparse
from utils.datasets import load_partitioned_datasets
from utils.training_utils import print_info, get_device
from utils.client_utils import client_fn, get_certificate
from utils.models import load_model_defination
import pdb, traceback




def main():
    parser = argparse.ArgumentParser(description='A description of your program')
    parser.add_argument('-a', '--server_address', type=str, default="[::]", help='Server address')
    parser.add_argument('-p', '--server_port', type=str, default="8080", help='Server port')
    parser.add_argument('-N', '--number_of_total_clients', type=int, default=2, help='Total number of clients')      
    parser.add_argument('-m', '--model_name', type=str, default = "basicCNN", help='Model name') 
    parser.add_argument('-d', '--dataset_name', type=str, default='CIFAR10', help='Dataset name') 
    parser.add_argument('-n', '--client_number', type=int, help='Client number')    
    parser.add_argument('-o', '--overfit_patience', type=int, default=-1, help='Patience after which to stop training, to prevent overfitting')
    parser.add_argument('-w', '--wandb_logging', action='store_true', help='Enable wandb logging')
    parser.add_argument('-db','--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-s', '--secure', action='store_true', help='Enable secure mode')
    parser.add_argument('-hl','--headless', action='store_true', help='Enable headless mode')
    args = parser.parse_args()
    model = load_model_defination(args.model_name, num_channels=3, num_classes=100)

    device = get_device()

    model.to(device)    
    
    if args.overfit_patience == -1:
        args.overfit_patience = 100000 # Initial patience, allow overfitting

    if args.secure:
        certificate = get_certificate()
    else:
        certificate = None
        
    print(f'certificate: {certificate}')
    
    
    print_info(device, args.model_name, args.dataset_name)

    trainloaders, valloaders, _, _ = load_partitioned_datasets(args.number_of_total_clients, dataset_name=args.dataset_name)
    client_defination = client_fn(args.client_number, model, trainloaders, valloaders, 
                                  args.number_of_total_clients, args.wandb_logging, args.dataset_name, 
                                  args.overfit_patience, simulation=args.headless
                                  )
    try:
        fl.client.start_numpy_client(
                                    server_address=args.server_address+':'+ args.server_port, 
                                    client=client_defination,
                                    root_certificates=certificate
                                    )
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