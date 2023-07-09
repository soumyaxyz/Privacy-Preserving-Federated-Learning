from torch.utils.data import DataLoader, random_split
from utils.datasets import *
from utils.training_utils import *

import pdb,traceback, wandb

def get_training_epoch():
    train_epochs = 1
    return train_epochs

def split_dataset(trainset, testset, num_clients: int, val_percent = 10, batch_size=32): 

    # Split training set into `num_clients` partitions to simulate different local datasets
    total_size = len(trainset)
    partition_size = total_size // num_clients
    lengths = [partition_size] * num_clients
    lengths[-1] += total_size% num_clients          # adding the reminder to the last partition

    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    val_datasets = []
    for ds in datasets:
        len_val = len(ds) // val_percent  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        try:            
            trainloaders.append(DataLoader(ds_train, batch_size, shuffle=True))
            valloaders.append(DataLoader(ds_val, batch_size))
        except Exception as e:
            traceback.print_exc()
            pdb.set_trace()

        
        val_datasets.append(ds_val)
    testloader = DataLoader(testset, batch_size)
    unsplit_valloader = DataLoader(torch.utils.data.ConcatDataset(val_datasets), batch_size)
    return trainloaders, valloaders, testloader, unsplit_valloader

def load_partitioned_datasets(num_clients: int, val_percent = 10, batch_size=32, dataset_name = 'CIFAR10'):
    trainset, testset = load_datasets(dataset_name)
    return split_dataset(trainset, testset, num_clients, val_percent, batch_size)



        
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, N , wandb_logging=False, dataset_name='CIFAR10'):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.wandb_logging = wandb_logging
        if self.wandb_logging:
            comment = 'Client_'+str(N)+'_'+net.__class__.__name__+'_'+dataset_name
            wandb_init(comment=comment, model_name=net.__class__.__name__, dataset_name=dataset_name)
    def __del__(self):
        if self.wandb_logging:
            wandb.finish()


    def get_parameters(self, config):
        # print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        # print(f"[Client {self.cid}] fit, config: {config}")
        # print(f"[Client {self.cid}] fit()")
        set_parameters(self.net, parameters)
        if get_training_epoch() == 1:
            self.loss, self.accuracy = train_single_epoch(self.net, self.trainloader)
        else:
            _, _, self.loss, self.accuracy = train(self.net, self.trainloader, self.valloader, epochs=get_training_epoch(),  wandb_logging=False, patience= 2)
        if self.wandb_logging:
            wandb.log({"train_acc": self.accuracy,"train_loss": self.loss})
            print(f"[Client {self.cid}] fit(loss: {self.loss:.4f}, accuracy: {self.accuracy:.4f})")
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        # print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        self.loss, self.accuracy = test(self.net, self.valloader)
        print(f"[Client {self.cid}] evaluate(loss: {self.loss:.4f}, accuracy: {self.accuracy:.4f})")
        return float(self.loss), len(self.valloader), {"accuracy": float(self.accuracy)}


def client_fn(cid, net, trainloaders, valloaders, N=2, wandb_logging=False, dataset_name='CIFAR10') -> FlowerClient:    
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader, N, wandb_logging, dataset_name)