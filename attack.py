import argparse
import pdb, traceback
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split 
from utils.models import load_model_defination
from utils.training_utils import get_device, train_shadow_model, get_device, print_info, save_model, test, load_model as load_saved_weights
from utils.datasets import get_datasets_details, load_datasets


class Membership_inference_attack:
    def __init__(self, target_model, target_dataset_name, shadow_model, shadow_count, attack_model, device, epochs, wandb_logging=False):
        
        self.shadow_count = shadow_count

        self.target_model           = target_model
        self.shadow_model           = shadow_model
        self.attack_model           = attack_model
        self.num_classes            = 10    
        trainset, testset           = load_datasets(target_dataset_name)
        self.datasets_details       = get_datasets_details(target_dataset_name)
        self.class_wise_trainset    = self.split_dataset_by_label(trainset)
        self.class_wise_testset     = self.split_dataset_by_label(testset)
        self.device                 = device
        self.epochs                 = epochs
        self.wandb_logging          = wandb_logging    

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
            attack_instance = Membership_inference_attack_instance( c,
                self.target_model,
                class_wise_datasets, 
                self.datasets_details,
                self.shadow_model, 
                self.shadow_count, 
                self.attack_model,
                self.device,
                self.epochs,
                self.wandb_logging
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
    def __init__(self, class_id, target_model, target_dataset, datasets_details, shadow_model_name, shadow_count, attack_model_name, device, epochs=50, wandb_logging=False):


        self.overlap_fraction   = 0.8
        self.seen_fraction      = 0.2 
        self.unseen_fraction    = 0.6
        self.val_fraction       = 0.1
        self.test_fraction      = 0.4
        self.batch_size         = 32
        
        self.shadow_count = shadow_count

        self.target_trainset, self.target_testset = target_dataset

        self.target_model       = target_model
        self.shadow_models      = []
        for i in range(shadow_count):
            self.shadow_models.append(load_model_defination(shadow_model_name, datasets_details).to(device))
        self.shadow_train, self.shadow_val, self.shadow_test    = self.get_shadow_datasets()
        self.attack_model       = load_model_defination(attack_model_name, datasets_details).to(device)
        self.epochs             = epochs
        self.wandb_logging      = wandb_logging
        self.shadow_models_trained = False

        
        
        print(f'Membership Inference Attack Instance on class {class_id} initialized')

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
        seen_size = int(trainset_size * self.seen_fraction) 
        seen_dataset =  Subset(trainset, torch.randperm(trainset_size)[:seen_size]) 
        
        testset_size = len(testset)
        unseen_size = int(testset_size * self.unseen_fraction)
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

        
        val_size    = int(len(target_dataset) * self.val_fraction)
        train_size  = len(target_dataset) - val_size

        

        attack_trainset, attack_valset = random_split(ConcatDataset(partial_dataset), [train_size, val_size], torch.Generator().manual_seed(42))
        
        attack_trainloder       = DataLoader(attack_trainset, self.batch_size, shuffle=True)
        attack_valloder         = DataLoader(attack_valset, self.batch_size)
        attack_testloder        = DataLoader(Loss_Label_Dataset(target_dataset, self.target_model), self.batch_size)

        return attack_trainloder, attack_valloder, attack_testloder


    def run_membership_inference_attack(self):
        
        # runs membership inference attack on the target model
        # creates a training loop where the output of the shadow model is trained to mimic the target model
        for i in range(self.shadow_count):
            train_shadow_model(self.target_model, self.shadow_models[i], self.shadow_train[i], self.shadow_val[i], self.epochs, verbose=False, wandb_logging=True)

        self.shadow_models_trained = True
        attack_trainloder, attack_valloder, attack_testloder = self.build_attack_dataset()

        self.train_attack_model(self.attack_model, attack_trainloder, attack_valloder, self.epochs, verbose=False, wandb_logging=True)

        loss, accurecy = self.test_attack_model(self.attack_model, attack_testloder)

        return loss, accurecy
    
    def train_attack_model(self, model, trainloader, valloader):
        return test(model, trainloader, valloader)


    def test_attack_model(self, model, testloader):
        return test(model, testloader)
    

def main():
    parser = argparse.ArgumentParser(description='A description of your program')
    parser.add_argument('-w', '--wandb_logging', action='store_true', help='Enable wandb logging')
    parser.add_argument('-e', '--num_epochs', type=int, default=50, help='Number of rounds of shadow training')
    parser.add_argument('-d', '--dataset_name', type=str, default='CIFAR10', help='Dataset name, target dataset ')
    parser.add_argument('-n', '--shadow_count', type=int, default=1, help='Number of shadow models')
    parser.add_argument('-m', '--target_model_name', type=str, default='basicCNN', help='Model name for the model to be attacked')
    parser.add_argument('-mw', '--target_model_weights', type=str, default='basicCNN', help='Weights for the model to be attacked')
    parser.add_argument('-s', '--shadow_model_name', type=str, default='basicCNN', help='Model name for the shadow model')
    parser.add_argument('-a', '--attack_model_name', type=str, default= 'attack_classifier', help='Classifier for the attack model')
    args = parser.parse_args()


    device = get_device()

    target_model        = load_model_defination(args.target_model_name, get_datasets_details(args.dataset_name)).to(device)
    load_saved_weights(target_model, filename =args.target_model_weights)

    attack = Membership_inference_attack(target_model, args.dataset_name, 
                                         args.shadow_model_name, 
                                         args.shadow_count, 
                                         args.attack_model_name, 
                                         device, args.num_epochs, args.wandb_logging
                                         )

    attack.start()




if __name__ == '__main__':
    main()