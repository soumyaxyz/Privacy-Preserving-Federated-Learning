import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader, random_split
import pdb,traceback


def load_CIFAR10():
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)

    return trainset, testset

def load_MNIST():
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )
    trainset = MNIST("./dataset", train=True, download=True, transform=transform)
    testset = MNIST("./dataset", train=False, download=True, transform=transform)

    return trainset, testset

def load_datasets(dataset_name = 'CIFAR10'):
    if dataset_name == 'CIFAR10':
        return load_CIFAR10()
    elif dataset_name == 'MNIST':
        return load_MNIST()
    else:
        # import pdb; pdb.set_trace()
        print(f'Unknown dataset name: {dataset_name}')
        raise NotImplementedError
    
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

def load_partitioned_datasets(num_clients: int, dataset_name = 'CIFAR10', val_percent = 10, batch_size=32):
    trainset, testset = load_datasets(dataset_name)
    return split_dataset(trainset, testset, num_clients, val_percent, batch_size)