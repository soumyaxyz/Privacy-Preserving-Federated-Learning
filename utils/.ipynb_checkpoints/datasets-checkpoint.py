
import cv2
import csv
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST, CIFAR100, SVHN, FashionMNIST
from torch.utils.data import  Dataset, DataLoader, ConcatDataset, Subset, TensorDataset, random_split
import pdb,traceback
from typing import List
import pandas as pd
import pprint
import matplotlib.pyplot as plt

from utils.lib import blockPrinting
from utils.cifar100_fine_coarse_labels import remapping
from utils.microsoft_preprocess import preprocess_microsoft_malware
from utils.training_utils import load_pickle, save_pickle
from utils.models import Load_LGB

# Dataset Classes definitions

class IncrementalDatasetWraper():
    def __init__(self, dataset_name = 'incremental_SVHN', audit_mode = False, addetive_train = False):
        self.name = dataset_name
        self.audit_mode = audit_mode
        self.splits = self._load_datasets(dataset_name)
        if addetive_train:
            self.splits = implement_addetive_dataset(self.splits, additive_train =True)

    @blockPrinting  
    def _load_datasets(self, dataset_name):
        if dataset_name == 'incremental_SVHN':
            data_splits = load_incremental_SVHN(uniform_test = True)
        elif dataset_name == 'incremental_test_SVHN':
            data_splits = load_incremental_SVHN(uniform_test = False)
        elif dataset_name == 'incremental_CIFAR100':
            data_splits = load_incremental_CIFAR100(remapping=[[5,6,7,8,9], [10,11,12,13,14], [15,16,17,18,19],[0,1,2,3,4]], uniform_test = True)
        elif dataset_name == 'incremental_test_CIFAR100':
            data_splits = load_incremental_CIFAR100(remapping=[[5,6,7,8,9], [10,11,12,13,14], [15,16,17,18,19],[0,1,2,3,4]], uniform_test = False)
        elif dataset_name == 'Microsoft_Malware_incremental':
            data_splits = load_incremental_Microsoft_Malware()
        else:
            print(f'Unknown dataset name: {dataset_name}')
            raise NotImplementedError
        
        for index, (train_subset, test_subset, num_channels, num_classes) in enumerate(data_splits):
            modified_trainset, modified_testset = remap_dataset(self.audit_mode, train_subset, test_subset)
            updated_split = (modified_trainset, modified_testset, num_channels, num_classes)
            data_splits[index] = updated_split

        return data_splits
        
    def select_split(self, split):
        self.trainset, self.testset, self.num_channels, self.num_classes = self.splits[split]
        self.data_split = [self.trainset, self.testset, self.num_channels, self.num_classes]


class DatasetWrapper():
    def __init__(self, dataset_name = 'CIFAR10', audit_mode = False):
        self.name = dataset_name
        self.audit_mode = audit_mode
        self.trainset, self.testset, self.num_channels, self.num_classes = self._load_datasets(dataset_name)

    def initilize_dataset_with_values(self, dataset_name, trainset, testset, num_channels, num_classes):
        self.name = dataset_name
        self.trainset = trainset
        self.testset = testset
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.audit_mode = None

    def get_X_y(self, val=True):
        X_train, y_train = extract_data_and_targets_with_dataloader(self.trainset)
        X_test, y_test = extract_data_and_targets_with_dataloader(self.testset)
        if val:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        else:
            X_val = X_test
            y_val = y_test
        return X_train, y_train, X_val, y_val, X_test, y_test


    #@blockPrinting  
    def _load_datasets(self, dataset_name):
        if dataset_name == 'CIFAR10':
            trainset, testset, num_channels, num_classes = load_CIFAR10()
        elif dataset_name == 'CIFAR100':
            trainset, testset, num_channels, num_classes =  load_CIFAR100()
        elif dataset_name == 'MNIST':
            trainset, testset, num_channels, num_classes =  load_MNIST()
        elif dataset_name == 'FashionMNIST':
            trainset, testset, num_channels, num_classes =  load_FashionMNIST()
        elif dataset_name == "SVHN":
            trainset, testset, num_channels, num_classes =  load_SVHN()
        elif dataset_name == 'Microsoft_Malware':
            trainset, testset, num_channels, num_classes =  load_Microsoft_Malware()
        elif 'incremental' in dataset_name:
            raise NotImplementedError('incremental dataset not implemented, use incrementalDatasetWraper() instead')            
        else:
            # import pdb; pdb.set_trace()
            print(f'Unknown dataset name: {dataset_name}')            
            raise NotImplementedError   
        
        #trainset, testset = remap_dataset(self.audit_mode, trainset, testset)

        return trainset, testset, num_channels, num_classes
        


class CIFAR_20_Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.remap = remapping()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # # Modify the label to be the new label based on modulo 10
        # mod_label = label % 10
        coarse_label = self.remap.fine_id_coarse_id[label]
        return img, coarse_label

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
            criterion = torch.nn.CrossEntropyLoss()


        if isinstance(self.target_model, torch.nn.Module):
            self.target_model.eval()
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
        elif isinstance(self.target_model, Load_LGB):

            X_test, y_test = extract_data_and_targets_with_dataloader(dataLoader.dataset) 
            overall_loss, _, y_pred = self.target_model.predict(X_test, y_test)

            if self.loss_batchwise:
                num_batches = len(y_test) // self.batch_size

                for i in range(num_batches):
                    start_index = i * self.batch_size
                    end_index = start_index + self.batch_size
                    y_test_batch = y_test[start_index:end_index]
                    y_pred_batch = y_pred[start_index:end_index] # type: ignore
                    
                    # Calculate MSE for the current batch and append to the list
                    mse = mean_squared_error(y_test_batch, y_pred_batch) # type: ignore
                    self.data.append(mse)
                    self.label.append(seen_unseen_label)
            else:
                squared_errors = (y_test - y_pred) ** 2
                for mse in squared_errors:
                    self.data.append(mse)
                    self.label.append(seen_unseen_label)
        else :
            raise Exception(f'Unknown model type: {type(self.target_model)}')

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

# Datasets functions

def load_CIFAR10():
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)

    num_channels=3
    num_classes=10


    #full_dataset = ConcatDataset([train_dataset,test_dataset])

    #train_size = int(len(full_dataset)*train_percent)
    #test_size = len(full_dataset) - train_size

    #trainset, testset = random_split(full_dataset, [train_size, test_size])




    return trainset, testset, num_channels, num_classes

def load_CIFAR100():
    # Download and transform CIFAR-100 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR100("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR100("./dataset", train=False, download=True, transform=transform)

    num_channels = 3
    num_classes = 100

    return trainset, testset, num_channels, num_classes

def load_SVHN():
    # Download and transform SVHN (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = SVHN("./dataset", split="train", download=True, transform=transform)
    testset = SVHN("./dataset", split="test", download=True, transform=transform)

    
    


    # Convert tensor to numpy array and change channel order for displaying
    


    num_channels=3
    num_classes = 10

    return trainset, testset, num_channels, num_classes

def load_MNIST():
    # Download and transform MNIST (train and test)
    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=3), #expand to 3 channels
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = MNIST("./dataset", train=True, download=True, transform=transform)
    testset = MNIST("./dataset", train=False, download=True, transform=transform)


    num_channels = 3
#     num_channels = 1
    num_classes = 10

    return trainset, testset, num_channels, num_classes

def load_FashionMNIST():
    # Download and transform FashionMNIST (train and test)
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),  # Expand to 3 channels
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    trainset = FashionMNIST("./dataset", train=True, download=True, transform=transform)
    testset = FashionMNIST("./dataset", train=False, download=True, transform=transform)

    num_channels = 3  # Set to 3 channels
    num_classes = 10

    return trainset, testset, num_channels, num_classes

def load_Microsoft_Malware():
    directory = 'dataset/MicrosoftMalware/'

    try:
        train_dataset, test_dataset, num_channels, num_classes =  load_pickle(directory+'saved_dataset.pkl')
    except:            
        train_dataset, test_dataset, num_channels, num_classes  =  preprocess_microsoft_malware(directory) 

        X_train, X_test, y_train, y_test = train_test_split(train_dataset, test_dataset, test_size=0.2, random_state=42)
        
        train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.long))

        try:
            save_pickle( (train_dataset, test_dataset, num_channels, num_classes), directory+'saved_dataset.pkl')
        except Exception as e:
            print('Error saving dataset:', e)
    
    return train_dataset, test_dataset, num_channels, num_classes
    
# Incremental Datasets wrappers functions

def load_incremental_Microsoft_Malware(num_splits = 4):
    data_splits = []

    train_dataset, test_dataset, num_channels, num_classes = load_Microsoft_Malware()

    total_size = len(train_dataset)
    partition_size = total_size // num_splits
    lengths = [partition_size] * num_splits
    lengths[-1] += total_size% num_splits          # adding the reminder to the last partition

    datasets = random_split(train_dataset, lengths, torch.Generator().manual_seed(42))

    for dataset in (datasets):

        data_splits.append([dataset, test_dataset, num_channels, num_classes])


    return data_splits
 
def load_incremental_SVHN(uniform_test):
    splits_paths=[
        './dataset/SVHN/extra_C',
        './dataset/SVHN/train_cropped_images',
        './dataset/SVHN/test_cropped_images',
        './dataset/SVHN/extra_A',
        './dataset/SVHN/extra_B'
    ]

    return load_incremental_local_dataset(splits_paths,uniform_test)

def load_incremental_local_dataset(splits_paths, uniform_test = True, combined_extra=False):
    data_splits = []
    print('Loading custom incremental dataset...')
    for directory in tqdm(splits_paths, leave=False):
        train_dataset, test_dataset, num_channels, num_classes = load_custom_image_dataset(directory, test_size=0.4)
        data_splits.append((train_dataset, test_dataset, num_channels, num_classes))

    if combined_extra:
        data_splits = combine_subsets(data_splits, [[0,1,2],3,4])

    #data_splits = implement_addetive_dataset(data_splits)
    
    data_splits = mix_subsets(data_splits)
    if uniform_test:
        data_splits = implement_combined_uniform_test(data_splits)
    else:
        data_splits = implement_addetive_dataset(data_splits)

    return data_splits

def load_incremental_CIFAR100(remapping= None, uniform_test = False):
    trainset, testset, num_channels, _ = load_CIFAR100()
    num_classes = 20
    # Split both train and test sets into 20 subsets
    train_subsets = split_dataset_into_subsets(trainset, num_classes)
    test_subsets = split_dataset_into_subsets(testset, num_classes)

    train_subsets = [CIFAR_20_Dataset(subset) for subset in train_subsets]
    test_subsets = [CIFAR_20_Dataset(subset) for subset in test_subsets]


    # Combine the train and test subsets along with num_channels and num_classes into a list of tuples
    data_splits = [(train_subsets[i], test_subsets[i], num_channels, num_classes) for i in range(len(train_subsets))]

    data_splits = mix_subsets(data_splits)
    if uniform_test:
        data_splits = implement_combined_uniform_test(data_splits)
    else:
        data_splits = implement_addetive_dataset(data_splits)


    if remapping is not None:
        data_splits = combine_subsets(data_splits, remapping)

    return data_splits


def remap_dataset(audit_mode, trainset, testset,  train_percent = 0.35, test_percent = 0.35 , audit_percent = 0.3, preserve_original_propertion = True):
    """
    Remaps the given train and test datasets based on the provided percentages. Holds out a portion of the training set for auditing purposes. 
    Depending on the wheather the audit_mode flag is set, the train and test sets are returned in different ways.

    Args:
    - trainset: The training dataset to be remapped.
    - testset: The testing dataset to be remapped.
    - train_percent: The percentage of samples to allocate to the training set (default is 0.35).
    - test_percent: The percentage of samples to allocate to the testing set (default is 0.35).
    - audit_percent: The percentage of samples to allocate to the audit set (default is 0.3).
    - preserve_original_propertion: A boolean indicating whether to preserve the original proportion of the training set overwriting the  train and test percentages (default is False).

    Returns:
    - train_set: The remapped training dataset.
    - test_set: The remapped testing dataset.
    """
    # Concatenate train and test sets
    full_dataset = ConcatDataset([trainset, testset])

    if preserve_original_propertion:        
        original_train_percentage = len(trainset) /len(full_dataset)
        train_total_percentage  = 1 - audit_percent        
        train_percent = original_train_percentage * train_total_percentage
        test_percent = (1-original_train_percentage) * train_total_percentage

    # Determine sizes of subsets
    num_samples = len(full_dataset)

    audit_train_percent = audit_percent *0.6  
    
    train_size = int(num_samples * train_percent)
    test_size = int(num_samples * test_percent)   
    audit_train_size = int(num_samples * audit_train_percent)
    audit_test_size = num_samples - (audit_train_size + train_size + test_size)       
        

    # Split the concatenated dataset into subsets
    train_set, test_set, audit_train_set, audit_test_set = random_split(full_dataset, [train_size, test_size, audit_train_size, audit_test_size], torch.Generator().manual_seed(42) )
    
        
    
    if audit_mode:
        train_set = audit_train_set
        test_set = audit_test_set
    return train_set, test_set

def combine_subsets(data_splits, subsets_groups):
    new_data_splits = []
    for group in subsets_groups:
        if isinstance(group, list):  # Group is a list of indices to combine
            train_datasets = [data_splits[i][0] for i in group]
            test_datasets = [data_splits[i][1] for i in group]
            # Assume num_channels and num_classes are consistent within the group
            num_channels = data_splits[group[-1]][2]
            num_classes = data_splits[group[-1]][3]
            combined_train_dataset = ConcatDataset(train_datasets)
            combined_test_dataset = ConcatDataset(test_datasets)
            new_data_splits.append((combined_train_dataset, combined_test_dataset, num_channels, num_classes))
        else:
            # Group is a single index, include as is
            new_data_splits.append(data_splits[group])
    return new_data_splits

def implement_addetive_dataset(data_splits, additive_train =False):
    new_data_splits = []
    expanding_dataset = []
    for i, split in tqdm(enumerate(data_splits), leave=False):
        train_dataset_i, test_dataset_i, num_channels, num_classes = split
        if additive_train:
            expanding_dataset.append(train_dataset_i)
            split = (ConcatDataset(expanding_dataset), test_dataset_i, num_channels, num_classes)
        else:
            expanding_dataset.append(test_dataset_i)
            split = (train_dataset_i, ConcatDataset(expanding_dataset), num_channels, num_classes)
        new_data_splits.append(split)
    return new_data_splits

def implement_combined_uniform_test(data_splits):    
    expanding_dataset = []
    for _, split in tqdm(enumerate(data_splits), leave=False):
        _, test_dataset_i, _, _ = split
        expanding_dataset.append(test_dataset_i)    
    combined_uniform_test= ConcatDataset(expanding_dataset)

    new_data_splits = []    
    for i, split in tqdm(enumerate(data_splits), leave=False):
        train_dataset_i, _, num_channels, num_classes = split
        split = (train_dataset_i, combined_uniform_test, num_channels, num_classes)
        new_data_splits.append(split)
    return new_data_splits

def extract_data_and_targets_with_dataloader(dataset):   
    loader = DataLoader(dataset, batch_size=64)    
    data_list = []
    targets_list = []    
    for data, targets in loader:
        data_list.append(data)
        targets_list.append(targets)        
    # Concatenate all batches
    data = torch.cat(data_list, dim=0)
    targets = torch.cat(targets_list, dim=0)    
    return data.numpy(), targets.numpy()

def load_custom_image_dataset(directory, test_size=0.4):
    images = []
    labels = []
    try:
        train_dataset, test_dataset, num_channels, num_classes =  load_pickle(directory+'.pkl')
    except:
        print(f'\nPresaved dataset not found, Loading custom dataset from {directory}')
        for label in tqdm(os.listdir(directory), leave=False):
            label_dir = os.path.join(directory, label)
            for img_file in tqdm(os.listdir(label_dir), leave=False):
                img_path = os.path.join(label_dir, img_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (32, 32))  # Resize image to a fixed size
                images.append(img)
                labels.append(int(label))
    
        # Transform the dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = [(transform(img), label) for img, label in zip(images, labels)]
    
        # Calculate the sizes of the train and test sets based on the test_size ratio
        test_size = int(len(dataset) * test_size)
        train_size = len(dataset) - test_size
    
        # Split the dataset into training and test sets
        torch.manual_seed(42)
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size]) # type: ignore
    
        num_channels = 3
        num_classes = len(np.unique(labels))

        try:
            save_pickle( (train_dataset, test_dataset, num_channels, num_classes), directory+'.pkl')
        except Exception as e:
            print('Error saving dataset:', e)


   
    return train_dataset, test_dataset, num_channels, num_classes

def split_dataset_into_subsets(dataset, num_subsets=10):
    # Assuming CIFAR-100 has 100 classes, grouped into 10 subsets
    class_groups = {k: [] for k in range(num_subsets)}  # Dict to hold subsets

    # Iterate through the dataset to group indices by class
    for idx, (_, label) in enumerate(dataset):
        # group_key = label // (100 // num_subsets)  # Determine the subset group
        group_key = label % num_subsets  # Group by modulo 10 of the label
        class_groups[group_key].append(idx)

    # Create a subset for each group
    subsets = [Subset(dataset, indices) for indices in class_groups.values()]
    return subsets

def get_mixing_proportions(num_classes = 20, seed_value=42):

    # Set the seed for reproducibility
    np.random.seed(seed_value)

    # Generate a random 10x10 matrix
    matrix = np.random.rand(num_classes, num_classes)

    # Normalize each row to sum up to 1
    matrix_normalized = matrix / matrix.sum(axis=1)[:, np.newaxis] 

    # Round the normalized matrix to 2 decimal places
    matrix_rounded = np.around(matrix_normalized, decimals=2)

    # Set the last column to 1 - the sum of the other columns, adjusting for rounding errors
    for row in matrix_rounded:
        row[-1] = 1 - row[:-1].sum()
        assert row.sum() == 1


    return matrix_rounded

def mix_subsets(subsets, proportions=None, seed_value=42):
    """
    Mix subsets according to user-defined proportions.

    Args:
    - subsets: A list of dataset subsets to mix.
    - proportions: A list of proportions for each subset.
    
    Returns:
    - A new dataset consisting of mixed subsets.
    """
    num_splits = len(subsets)
    if proportions is None:
        proportions = get_mixing_proportions(num_splits, seed_value)
    else:
        assert num_splits == len(proportions)

    # Initialize empty lists for the new subsets
    new_train_datasets = [[] for _ in range(num_splits)]
    new_test_datasets = [[] for _ in range(num_splits)]
    new_data_splits = []

    generator = torch.Generator().manual_seed(seed_value)


    try:

        # Loop through each original subset
        for i, subset in enumerate(subsets):
            trainset_i, testset_i, num_channels, num_classes = subset
            
            # Calculate lengths for the new subsets
            lengths_train = [int(p * len(trainset_i)) for p in proportions[i]]
            lengths_test = [int(p * len(testset_i)) for p in proportions[i]]
            

            #fix for rounding errors
            lengths_train[-1] = len(trainset_i)-sum(lengths_train[:-1]) 
            lengths_test[-1] = len(testset_i)-sum(lengths_test[:-1])

            # Split the original subsets into new subsets based on the calculated lengths
            train_splits = random_split(trainset_i, lengths_train, generator=generator)
            test_splits = random_split(testset_i, lengths_test, generator=generator)

            # Accumulate the splits into the corresponding new datasets arrays
            for i, split in enumerate(train_splits):
                new_train_datasets[i].append(split)  
            
            for i, split in enumerate(test_splits):
                new_test_datasets[i].append(split)  

        # Now, concatenate the accumulated subsets
        for i, (trn_datasets, tst_datasets) in enumerate(zip(new_train_datasets, new_test_datasets)):
            split = (ConcatDataset(trn_datasets), ConcatDataset(tst_datasets), num_channels, num_classes)
            new_data_splits.append(split)
    except Exception as e:
        traceback.print_exc()
        pdb.set_trace()

    return new_data_splits

def load_loss_dataset(filename='dataset'):
    print(f'\tLoading dataset from {filename}')
    load_path = './saved_models/' + filename + '.csv'
    dataset = []

    with open(load_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        data = []
        label = []

        for row in reader:
            data_i, label_i = row
            # Convert data and label to appropriate types if needed
            try:
                data.append(eval(data_i))
                label.append(eval(label_i))
            except:
                traceback.print_exc()
                # pdb.set_trace()

            

        dataset = Wrapper_Dataset(data, label)

    return dataset

def split_dataloaders(trainset, testset, num_splits: int, split_test = False, val_percent = 10, batch_size=32):#-> tuple[List, List, DataLoader, DataLoader]: 
    

    # Split training set into `num_clients` partitions to simulate different local datasets
    total_size = len(trainset)
    partition_size = total_size // num_splits
    lengths = [partition_size] * num_splits
    lengths[-1] += total_size% num_splits          # adding the reminder to the last partition

    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    val_datasets = []
    for ds in datasets:
        if val_percent == 0:
            len_val = 0
        else:
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
    if split_test:
        total_size = len(testset)
        partition_size = total_size // num_splits
        lengths = [partition_size] * num_splits
        lengths[-1] += total_size% num_splits          # adding the reminder to the last partition

        datasets = random_split(testset, lengths, torch.Generator().manual_seed(42))
        testloaders = []
        for ds in datasets:
            testloaders.append(DataLoader(ds, batch_size))
        unsplit_valloader = None
    else: 
        testloader = DataLoader(testset, batch_size)
        unsplit_valloader = DataLoader(torch.utils.data.ConcatDataset(val_datasets), batch_size) #type:ignore

    return trainloaders, valloaders, testloader, unsplit_valloader #type:ignore

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

def load_partitioned_continous_datasets(num_clients, dataset_split, val_percent = 10, batch_size=32):# -> tuple[tuple, int, int]:
    [train_dataset, test_dataset, num_channels, num_classes] = dataset_split   

    return split_dataloaders(train_dataset, test_dataset, num_clients, split_test=False,val_percent=val_percent, batch_size=batch_size), num_channels, num_classes 

def load_dataset(dataset_name = 'CIFAR10'):
    return DatasetWrapper(dataset_name)

def load_partitioned_dataloaders(num_clients: int, dataset_name = 'CIFAR10', val_percent = 10, batch_size=32): # -> tuple[tuple, int, int]:  
    dataset = load_dataset(dataset_name)    
    return split_dataloaders(dataset.trainset, dataset.testset, num_clients, split_test=False,val_percent=val_percent, batch_size=batch_size), dataset.num_channels, dataset.num_classes

