import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST, CIFAR100, SVHN, FashionMNIST
from torch.utils.data import  Dataset, DataLoader, ConcatDataset, random_split
from utilities.lib import blockPrinting
import pdb,traceback
from typing import List


class DatasetWrapper():
    def __init__(self, dataset_name = 'CIFAR10', data_path="~/dataset"):
        
        self.data_path = data_path
        self.name = dataset_name
        self.trainset, self.testset, self.num_channels, self.num_classes = self._load_datasets(dataset_name)


    # @blockPrinting  
    def _load_datasets(self, dataset_name):
        if dataset_name == 'CIFAR10':
            return load_CIFAR10(self.data_path)
        elif dataset_name == 'CIFAR100':
            return load_CIFAR100(self.data_path)
        elif dataset_name == 'MNIST':
            return load_MNIST(self.data_path)
        elif dataset_name == 'FashionMNIST':
            return load_FashionMNIST(self.data_path)
        elif dataset_name == "SVHN":
            return load_SVHN(self.data_path)
        else:
            # import pdb; pdb.set_trace()
            print(f'Unknown dataset name: {dataset_name}')
            raise NotImplementedError
    
   


def load_CIFAR10(data_path="~/dataset"):
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(root=data_path, train=True, download=True, transform=transform)
    testset = CIFAR10(root=data_path, train=False, download=True, transform=transform)

    num_channels=3
    num_classes=10


    #full_dataset = ConcatDataset([train_dataset,test_dataset])

    #train_size = int(len(full_dataset)*train_percent)
    #test_size = len(full_dataset) - train_size

    #trainset, testset = random_split(full_dataset, [train_size, test_size])




    return trainset, testset, num_channels, num_classes

def load_CIFAR100(data_path="~/dataset"):
    # Download and transform CIFAR-100 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR100(root=data_path, train=True, download=True, transform=transform)
    testset = CIFAR100(root=data_path, train=False, download=True, transform=transform)

    num_channels = 3
    num_classes = 100

    return trainset, testset, num_channels, num_classes

def load_SVHN(data_path="~/dataset"):
    # Download and transform SVHN (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = SVHN(root=data_path, split="train", download=True, transform=transform)
    testset = SVHN(root=data_path, split="test", download=True, transform=transform)

    num_channels=3
    num_classes = 10

    return trainset, testset, num_channels, num_classes

def load_MNIST(data_path="~/dataset"):
    # Download and transform MNIST (train and test)
    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=3), #expand to 3 channels
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = MNIST(root=data_path, train=True, download=True, transform=transform)
    testset = MNIST(root=data_path, train=False, download=True, transform=transform)


    num_channels = 3
#     num_channels = 1
    num_classes = 10

    return trainset, testset, num_channels, num_classes


def load_FashionMNIST(data_path="~/dataset"):
    # Download and transform FashionMNIST (train and test)
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),  # Expand to 3 channels
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    trainset = FashionMNIST(root=data_path, train=True, download=True, transform=transform)
    testset = FashionMNIST(root=data_path, train=False, download=True, transform=transform)

    num_channels = 3  # Set to 3 channels
    num_classes = 10

    return trainset, testset, num_channels, num_classes






class Loss_Label_Dataset(Dataset):
    """Loss_label_Dataset."""

    def __init__(self, original_dataset, target_model, device, batch_size = 32, loss_batchwise = False):
        self.batch_size         = batch_size  
        self.loss_batchwise     = loss_batchwise
        trainset                = original_dataset[0]
        testset                 = original_dataset[1]
        seen_count              = trainset.dataset.__len__()
        unseen_count            = testset.dataset.__len__()
        self.target_model       = target_model
        self.device             = device

        try:
            assert abs(seen_count - unseen_count) < seen_count/10  # roughly ballanced dataset
            # print(f'Ballanced dataset: seen {seen_count}, unseen {unseen_count}')
        except AssertionError as e:
            type  = 'batchwise' if loss_batchwise else 'samplewise'
            print(f'\tUnballanced {type} dataset: seen {seen_count}, unseen {unseen_count}')
            # pdb.set_trace()

        self.data   = []
        self.label  = []

        self.append_data_label(trainset, 1.0)
        self.append_data_label(testset, 0.0)

        # pdb.set_trace()
        

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):        
        sample = [self.data[idx], self.label[idx]]
        return sample
    
    def append_data_label(self, dataLoader, seen_unseen_label, criterion=None):
        if not criterion:
            criterion = torch.nn.CrossEntropyLoss( )


        for images, labels in dataLoader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.target_model(images)
            if self.loss_batchwise:
                loss = criterion(outputs, labels).item()
                self.data.append(loss)
                self.label.append(seen_unseen_label)               

            else:
                for i, label in enumerate(labels):
                    instance_loss = criterion(outputs[i], label).item()
                    self.data.append(instance_loss)
                    self.label.append(seen_unseen_label)

        return 


class Wrapper_Dataset(Dataset):
    def __init__(self, data, label):
        self.data   = data
        self.label  = label
         
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):        
        sample = [self.data[idx], self.label[idx]]
        return sample

    
class Error_Label_Dataset(Loss_Label_Dataset):
    def __init__(self, original_dataset, target_model, device, batch_size=32):
        super().__init__(original_dataset, target_model, device, batch_size)

    def append_data_label(self, dataLoader, seen_unseen_label, criterion=None):
        if not criterion:
            criterion = torch.nn.CrossEntropyLoss()


        for images, _ in dataLoader:
            images  = images.to(self.device)
            outputs = self.target_model(images)           

            # pdb.set_trace()

            self.data.append(outputs)
            self.label.append(seen_unseen_label)

        return 



def split_dataset(trainset, testset, num_clients: int, val_percent = 10, batch_size=32)-> tuple[List, List, DataLoader, DataLoader]: 


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
    unsplit_valloader = DataLoader(torch.utils.data.ConcatDataset(val_datasets), batch_size) #type:ignore
    return trainloaders, valloaders, testloader, unsplit_valloader

def load_dataloaders(trainset, testset, batch_size=32):
    trainloader    = DataLoader(trainset, batch_size, shuffle=True)
    testloader     = DataLoader(testset, batch_size)
    return trainloader,  testloader

def get_dataloaders_subset(dataloader, random_subset_size):
    dataset  = dataloader.dataset
    lengths  = [random_subset_size, len(dataset) - random_subset_size]
    truncated_dataset = random_split(dataset, lengths, torch.Generator().manual_seed(42))
    return DataLoader(truncated_dataset[0], dataloader.batch_size, shuffle=True)

def merge_dataloaders(trainloaders):    
    trn_datasets = []
    for loader in trainloaders:
        trn_datasets.append(loader.dataset)
    return DataLoader(ConcatDataset(trn_datasets), trainloaders[0].batch_size)


def load_partitioned_datasets(num_clients: int, dataset_name = 'CIFAR10', data_path="~/datasets", val_percent = 10, batch_size=32, split=None) -> tuple[tuple, int, int]:
    if split is not None:
        dataset = DatasetWrapper(dataset_name, data_path)
    else:
        dataset = IncrementalDatasetWrapper(dataset_name, data_path)
        dataset = dataset.get_split(num_clients, split)
    return split_dataset(dataset.trainset, dataset.testset, num_clients, val_percent, batch_size), dataset.num_channels, dataset.num_classes