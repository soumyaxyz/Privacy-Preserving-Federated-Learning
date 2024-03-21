from time import sleep
import flwr as fl
import argparse
import torch
from utils.datasets import ContinuousDatasetWraper, load_partitioned_continous_datasets, load_partitioned_datasets
from utils.training_utils import print_info, get_device
from utils.client_utils import client_fn, get_certificate
from utils.models import load_model_defination
import pdb, traceback

def wait_and_retry(server_address, client_definition,certificate, debug=False, tries=10):
    print('Server unavailable. Retrying in 5 seconds...')
    sleep(5)
    attempts = 1   
    while attempts<=tries:
        try:
            fl.client.start_numpy_client(server_address=server_address, 
                                         client=client_definition,
                                         root_certificates=certificate)
            attempts = tries+1  # Exit loop after successful connection
        except KeyboardInterrupt:
            break  # Exit loop on CTRL+C
        except Exception as e:
            if 'UNAVAILABLE' in str(e):
                print('Server unavailable. Retrying in 5 seconds...')
                sleep(5)  # Wait before retrying
                attempts += 1
            else:
                if debug:
                    traceback.print_exc()
                    pdb.set_trace()
                else:
                    print("Stopped with errors. Exiting.")
                attempts = tries+1  # Do not retry for non-UNAVAILABLE errors




def run_client_once(args, model, trainloaders, valloaders, optimizer):
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

    
    client_definition = client_fn(args.client_number, model, trainloaders, valloaders, 
                                  optimizer=optimizer, 
                                  N=args.number_of_total_clients, 
                                  wandb_logging=args.wandb_logging, 
                                  dataset_name=args.dataset_name, 
                                  differential_privacy=args.differential_privacy,
                                  patience=args.overfit_patience, 
                                  simulation=args.headless)
    
    
    
    server_address=args.server_address+':'+ args.server_port 

    try:
        fl.client.start_numpy_client(
                                    server_address=server_address, 
                                    client=client_definition,
                                    root_certificates=certificate
                                    )
    # except WSA Error
    except KeyboardInterrupt:
        print("Stopped with by user. Exiting.")
    except Exception as e:
        error_message = str(e)
        if 'UNAVAILABLE' in error_message: # 'UNAVAILABLE'
            wait_and_retry(server_address, client_definition,certificate, args.debug)  
        elif args.debug:
            traceback.print_exc()
            pdb.set_trace()
        else:
            print("Stopped with errors. Exiting.")


def main():
    parser = argparse.ArgumentParser(description='A description of your program')
    parser.add_argument('-a', '--server_address', type=str, default="localhost", help='Server address')
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
    parser.add_argument('-dp', '--differential_privacy', action='store_true', help='Enable differential privacy')
    args = parser.parse_args()
    
    if 'continuous' in args.dataset_name: 
        continous_datasets = ContinuousDatasetWraper(args.dataset_name)

        _, _, num_channels, num_classes = continous_datasets.splits[0]
        model = load_model_defination(args.model_name, num_channels, num_classes, args.differential_privacy)

        optimizer = torch.optim.Adam(model.parameters())
        

        for i,dataset_split in enumerate(continous_datasets.splits):
            [trainloaders, valloaders, _, _], num_channels_i, num_classes_i = load_partitioned_continous_datasets(num_clients=args.number_of_total_clients, dataset_split=dataset_split)
            assert num_channels_i == num_channels
            assert num_classes_i == num_classes            
            
            run_client_once(args, model, trainloaders, valloaders, optimizer) 

            print(f'\n\nDone with data split {i}\n\n')
    else:
        [trainloaders, valloaders, _, _], num_channels, num_classes = load_partitioned_datasets(args.number_of_total_clients, dataset_name=args.dataset_name)
        model = load_model_defination(args.model_name, num_channels=num_channels, num_classes=num_classes, differential_privacy =args.differential_privacy)
        optimizer = torch.optim.Adam(model.parameters())

        run_client_once(args, model, trainloaders, valloaders, optimizer)
    



if __name__ == '__main__':
    main()