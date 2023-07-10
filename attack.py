import pdb, traceback
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split 
from torchvision.datasets import Dataset
from utils.training_utils import get_device, train_shadow_model, train_attack_model
from utils.datasets import load_datasets


class Membership_inference_attack:
    def __init__(self, target_model, target_dataset, shadow_model, shadow_count):
        
        self.shadow_count = shadow_count

        self.target_model           = target_model
        self.target_dataset         = target_dataset 
        self.shadow_model           = shadow_model
        self.num_classes            = 10    
        trainset, testset           = load_datasets('CIFAR10')
        self.class_wise_trainset    = self.split_dataset_by_label(trainset)
        self.class_wise_testset     = self.split_dataset_by_label(testset)
        self.epochs                 = 50    

    def split_dataset_by_label(self, dataset):   
        class_datasets = [[] for _ in range(self.num_classes)]      

        # Assign samples to their respective class lists
        for sample, label in dataset:
            class_datasets[label].append((sample, label))

        return class_datasets
    
    def start(self):
        loss, accuracy = 0.0, 0.0
        for c in range(self.num_classes):
            class_wise_datasets = [self.class_wise_trainset[c], self.class_wise_testset[c]]
            attack_instance = Membership_inference_attack_instance(
                self.target_model,
                class_wise_datasets, 
                self.target_dataset, 
                self.shadow_model, 
                self.shadow_count, 
                self.epochs
            )
            class_loss, class_accuracy = attack_instance.run_membership_inference_attack()
            loss        += class_loss
            accuracy    += class_accuracy

        print(f"Overall Loss: {loss}, and Overall Accuracy: {accuracy}")
        


class Loss_Label_Dataset(Dataset):
    """Loss_label_Dataset."""

    def __init__(self, original_dataset, target_model):
       
        trainset                = original_dataset[0]
        testset                 = original_dataset[1]
        self.target_model       = target_model
        self.device             = get_device()

        self.data = []
        self.label = []
        self.append_loss_label(trainset, 1)
        self.append_loss_label(testset, 0)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):        
        sample = {'loss': self.data[idx], 'label': self.label[idx]}
        return sample
    
    def append_loss_label(self, dataset, seen_unseen_label, criterion=None):
        if not criterion:
            criterion = torch.nn.CrossEntropyLoss()

        for images, labels in dataset:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.target_model(images)
            loss = criterion(outputs, labels).item()

            self.data.append(loss)
            self.label.append(seen_unseen_label)
        return 

    


class Membership_inference_attack_instance:
    def __init__(self, target_model, target_dataset, shadow_model, shadow_count,epochs=50):
        
        self.shadow_count = shadow_count

        self.target_trainset, self.target_testset = target_dataset

        self.target_model       = target_model
        self.shadow_models      = [shadow_model]*shadow_count
        self.shadow_train, self.shadow_val, self.shadow_test    = self.get_shadow_datasets()
        self.attack_model       = get_attack_model()
        self.epochs             = epochs
        self.shadow_models_trained = False

        
        self.overlap_fraction=.8
        self.seen_fraction=.2, 
        self.unseen_fraction=.6
        self.val_fraction=.1
        self.test_fraction=.4,
        self.batch_size = 32

    def get_shadow_datasets(self):
        initial_dataset = self.build_master_shadow_dataset(self.target_trainset, self.target_testset)
        return self.get_final_shadow_datasets(initial_dataset)
        
    def get_final_shadow_datasets(self, initial_dataset ):
        # Split dataset into `self.shadow_count` overlapping partitions, with 'overlap_fraction' overlap, to simulate different shadow datasets
        total_size = len(initial_dataset)
        partition_size = int(total_size*self.overlap_fraction)        

        datasets = []
        for i in range(self.shadow_count):
            datasets.append( Subset(initial_dataset, torch.randperm(total_size)[:partition_size]) )


        # Split each partition into train/val/test and create DataLoader
        trainloaders = []
        valloaders = []
        testloaders = []
        for ds in datasets:
            len_test    = int(len(ds) * self.test_fraction)      # test set size
            len_val     = int(len(ds) *self.val_fraction)        # validation set size
            len_train = len(ds) - (len_test + len_val)      # train set size
            lengths = [len_train, len_val, len_test]
            ds_train, ds_val, ds_test = random_split(ds, lengths, torch.Generator().manual_seed(42))
            try:
                trainloaders.append(DataLoader(ds_train, self.batch_size, shuffle=True))
                valloaders.append(DataLoader(ds_val, self.batch_size))
                testloaders.append(DataLoader(ds_test, self.batch_size))
            except Exception as e:
                traceback.print_exc()
                pdb.set_trace()
            
        return trainloaders, valloaders, testloaders
    
    def build_master_shadow_dataset(self, trainset, testset):
        
        trainset_size = len(trainset)
        seen_size = trainset_size * self.seen_fraction 
        seen_dataset =  Subset(trainset, torch.randperm(trainset_size)[:seen_size]) 
        
        testset_size = len(testset)
        unseen_size = testset_size * self.unseen_fraction 
        unseen_dataset =  Subset(trainset, torch.randperm(testset_size)[:unseen_size])

        return ConcatDataset([seen_dataset, unseen_dataset])

    
    def build_attack_dataset(self):
        if not self.shadow_models_trained:
            raise Exception("Shadow models not trained")
        partial_dataset = []
        for i in range(self.shadow_count):
            shadow_model    = self.shadow_models[i]
            shadow_dataset  = [self.shadow_train[i], self.shadow_val[i]]
            partial_dataset.append(  Loss_Label_Dataset(shadow_dataset, shadow_model) )
        
        target_dataset= [self.target_trainset, self.target_testset]

        attack_trainset, attack_valset = random_split(ConcatDataset(partial_dataset), lengths, torch.Generator().manual_seed(42))
        
        attack_trainloder   = DataLoader(attack_trainset, self.batch_size, shuffle=True)
        attack_valloder      = DataLoader(attack_valset, self.batch_size)
        attack_testloder    = DataLoader(Loss_Label_Dataset(target_dataset, self.target_model), self.batch_size)

        return attack_trainloder, attack_valloder, attack_testloder


    def run_membership_inference_attack(self):
        
        # runs membership inference attack on the target model
        # creates a training loop where the output of the shadow model is trained to mimic the target model
        for i in range(self.shadow_count):
            train_shadow_model(self.target_model, self.shadow_models[i], self.shadow_train[i], self.shadow_val[i], self.epochs, verbose=False, wandb_logging=True)

        self.shadow_models_trained = True
        attack_trainloder, attack_valloder, attack_testloder = self.build_attack_dataset()

        train_attack_model(self.attack_model, attack_trainloder, attack_valloder, self.epochs, verbose=False, wandb_logging=True)

        loss, accurecy = test_attack_model(self.attack_model, attack_testloder)

        return loss, accurecy