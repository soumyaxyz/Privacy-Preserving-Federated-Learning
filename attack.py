import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split 
from torchvision.datasets import load_datasets
from utils.training_utils import *


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
    
    def start():
        for c in range(self.num_classes):
            self.class_wise_dataset[c] 
            attack_instance = Membership_inference_attack_instance(
                self.target_model, 
                self.target_dataset, 
                self.shadow_model, 
                self.shadow_count, 
                self.class_wise_trainset[c], 
                self.class_wise_testset[c],
                self.epochs
            )
            attack_instance.run_membership_inference_attack()




    


class Membership_inference_attack_instance:
    def __init__(self, target_model, target_dataset, shadow_model, shadow_count, trainset, testset, epochs=50):
        
        self.shadow_count = shadow_count

        

        self.target_model       = target_model
        # self.target_dataset     = target_dataset
        self.shadow_models      = [shadow_model]*shadow_count
        self.shadow_train, self.shadow_val, self.shadow_test    = self.get_shadow_datasets(class_datasets)
        self.attack_model       = get_attack_model()
        self.attack_dataset     = self.get_attack_dataset()
        self.epochs             = epochs

        
        self.overlap_fraction=.8
        self.seen_fraction=.2, 
        self.unseen_fraction=.6
        self.val_fraction=.1
        self.test_fraction=.4,
        self.batch_size = 32

    def get_shadow_datasets(self, trainset, testset):
        initial_dataset = self.build_master_shadow_dataset(trainset, testset)
        return self.get_final_shadow_datasets(self, initial_dataset)
        
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

    
    def build_master_shadow_dataset(self,trainset, testset):
        
        trainset_size = len(trainset)
        seen_size = trainset_size * self.seen_fraction 
        seen_dataset =  Subset(trainset, torch.randperm(trainset_size)[:seen_size]) 
        
        testset_size = len(testset)
        unseen_size = testset_size * self.unseen_fraction 
        unseen_dataset =  Subset(trainset, torch.randperm(testset_size)[:unseen_size])

        return ConcatDataset([seen_dataset, unseen_dataset])
            
    def run_membership_inference_attack(self):
        
        # runs membership inference attack on the target model
        # creates a training loop where the output of the shadow model is trained to mimic the target model
        for i in range(self.shadow_count):
            train_shadow_model(self.target_model, self.shadow_models[i], self.shadow_train[i], self.shadow_val[i], self.epochs, verbose=False, wandb_logging=True)

        for i in range(self.shadow_count):
            train_attack_model(self.attack_model, self.attack_dataset, self.shadow_models[i], self.shadow_train[i], self.shadow_val[i], self.epochs, verbose=False, wandb_logging=True)