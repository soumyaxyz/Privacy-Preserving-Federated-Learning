from utils.client_utils import FlowerClient, load_partitioned_datasets, print_info, get_device




def main():
    parser = argparse.ArgumentParser(description='A description of your program')
    parser.add_argument('-a', '--server_address', type=str, default="[::]", help='Server address')
    parser.add_argument('-p', '--server_port', type=str, default="8080", help='Server port')
    parser.add_argument('-N', '--number_of_total_clients', type=int, default=2, help='Total number of clients')      
    parser.add_argument('-m', '--model_name', type=str, default = "basicCNN", help=Model name')  
    parser.add_argument('-n', '--client_number', type=int, help='Client number')
    args = parser.parse_args()
    model = load_model(args.model_name, num_channels=3, num_classes=10) 
    trainloaders, valloaders, _, _ = load_partitioned_datasets(args.number_of_total_clients)
    client_defination = client_fn(args.client_number, model, trainloaders, valloaders)
    fl.client.start_numpy_client(server_address=args.server_address+':'+ args.server_port, client=client_defination)



if __name__ == '__main__':
    main()