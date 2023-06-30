import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from utils.training_utils import *
from utils.models import basicCNN as Net

import pdb,traceback


DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)


def load_CIFAR10():
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)

    return trainset, testset


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
    val_dataset = []
    for ds in datasets:
        len_val = len(ds) // val_percent  # 10 % validation set
        len_train = len(ds) - len_val
        sun_lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, sun_lengths, torch.Generator().manual_seed(42))
        try:            
            trainloaders.append(DataLoader(ds_train, batch_size, shuffle=True))
            valloaders.append(DataLoader(ds_val, batch_size))
        except Exception as e:
            traceback.print_exc()
            pdb.set_trace()

        
        val_dataset.append(ds_val)
    testloader = DataLoader(testset, batch_size)
    unsplit_valloader = DataLoader(torch.utils.data.ConcatDataset(val_dataset), batch_size)
    return trainloaders, valloaders, testloader, unsplit_valloader

def load_datasets(num_clients: int, val_percent = 10, batch_size=32, dataset_name = 'CIFAR10'):
    if dataset_name == 'CIFAR10':
        trainset, testset = load_CIFAR10()
        # pdb.set_trace()
        return split_dataset(trainset, testset, num_clients, val_percent, batch_size)
    else:
        raise NotImplementedError


        
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        # print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        # print(f"[Client {self.cid}] fit, config: {config}")
        set_parameters(self.net, parameters)
        train_single_epoch(self.net, self.trainloader)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        # print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}




class Client_function_wrapper_class():
    """docstring for ClassName"""
    def __init__(self, trainloaders, valloaders):
        # super(ClassName, self).__init__()
        self.trainloaders = trainloaders
        self.valloaders = valloaders

    def client_fn(self, cid) -> FlowerClient:
        net = Net().to(DEVICE)
        trainloader = self.trainloaders[int(cid)]
        valloader = self.valloaders[int(cid)]
        return FlowerClient(cid, net, trainloader, valloader)