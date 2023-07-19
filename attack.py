import argparse
import pdb, traceback
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split 
from utils.models import load_model_defination
from utils.training_utils import get_device, save_loss_dataset, train_shadow_model, get_device, print_info, save_model,train, test, load_model as load_saved_weights
from utils.datasets import get_datasets_details, load_datasets, load_dataloaders


class Membership_inference_attack:
    def __init__(self, 
                 target_model, target_dataset_name, 
                 shadow_model_name, shadow_count, 
                 attack_model_name, 
                 device, 
                 shadow_epochs, attack_epochs, 
                 wandb_logging=False
                 ):
        
        self.shadow_count = shadow_count

        self.target_model           = target_model
        self.shadow_model_name      = shadow_model_name
        self.attack_model_name      = attack_model_name
        self.num_classes            = 10    
        trainset, testset           = load_datasets(target_dataset_name)
        self.datasets_details       = get_datasets_details(target_dataset_name)
        self.class_wise_trainset    = self.split_dataset_by_label(trainset)
        self.class_wise_testset     = self.split_dataset_by_label(testset)
        self.device                 = device
        self.shadow_epochs          = shadow_epochs
        self.attack_epochs          = attack_epochs
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
            print(f'\nMembership Inference Attack Instance on class {c} initialized')
            class_wise_datasets = [self.class_wise_trainset[c], self.class_wise_testset[c]]
            attack_instance = Membership_inference_attack_instance( c,
                self.target_model,
                class_wise_datasets, 
                self.datasets_details,
                self.shadow_model_name, 
                self.shadow_count, 
                self.attack_model_name,
                self.device,
                self.shadow_epochs,
                self.attack_epochs,
                self.wandb_logging
            )
            class_loss, class_accuracy = attack_instance.run_membership_inference_attack()
            print(f"Loss on class {c}: {class_loss}, and Accuracy on class {c}: {class_accuracy}")
            loss        += class_loss
            accuracy    += class_accuracy
        
        loss     /= self.num_classes
        accuracy /= self.num_classes

        print(f"\nOverall Loss: {loss}, and Overall Accuracy: {accuracy}")
        


class Loss_Label_Dataset(Dataset):
    """Loss_label_Dataset."""

    def __init__(self, original_dataset, target_model, batch_size = 32):
        self.batch_size         = batch_size 
        # trainset, testset     = load_dataloaders(original_dataset[0], original_dataset[1], batch_size)   
        trainset                = original_dataset[0]
        testset                 = original_dataset[1]
        seen_count              = len(trainset)
        unseen_count            = len(testset)
        self.target_model       = target_model
        self.device             = get_device()

        try:
            assert abs(seen_count - unseen_count) < seen_count/10  # roughly ballanced dataset
            # print(f'Ballanced dataset: seen {seen_count}, unseen {unseen_count}')
        except AssertionError as e:
            print(f'Unballanced dataset: seen {seen_count}, unseen {unseen_count}')
            # pdb.set_trace()

        self.data   = []
        self.label  = []
        seen_count   = self.append_loss_label(trainset, 1.0)
        unseen_count = self.append_loss_label(testset, 0.0)
        

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):        
        sample = [self.data[idx], self.label[idx]]
        return sample
    
    def append_loss_label(self, dataLoader, seen_unseen_label, criterion=None):
        if not criterion:
            criterion = torch.nn.CrossEntropyLoss()


        for images, labels in dataLoader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.target_model(images)
            loss = criterion(outputs, labels).item()

            # pdb.set_trace()

            self.data.append(loss)
            self.label.append(seen_unseen_label)

        return 

def build_label_dict(ds):
    bal_dict = dict()
    bal_dict[0] = 0
    bal_dict[1] = 0
    for j in range(ds.__len__()):
        _, lbl = ds.__getitem__(j)
        bal_dict[lbl] += 1
    print(f'{bal_dict}')


class Membership_inference_attack_instance:
    def __init__(self, class_id, 
                 target_model, target_dataset, 
                 datasets_details, 
                 shadow_model_name, shadow_count, 
                 attack_model_name, 
                 device, 
                 shadow_epochs=50, 
                 attack_epochs=50, 
                 wandb_logging=False
                 ):

        self.class_id               = class_id
        self.overlap_fraction       = 0.8
        self.seen_fraction          = 0.12 
        self.unseen_fraction        = 0.6
        self.val_fraction           = 0.2
        self.shadow_test_fraction   = 0.5
        self.batch_size             = 32
        self.device                 = device
        
        self.shadow_count = shadow_count

        self.target_trainset, self.target_testset = target_dataset    
        num_channels, num_classes = datasets_details

        self.target_model       = target_model
        self.shadow_models      = []
        for _ in range(shadow_count):
            self.shadow_models.append(load_model_defination(shadow_model_name, num_channels, num_classes).to(self.device))
            
            
        self.shadow_train_dataloader, self.shadow_test_dataloader    = self.get_shadow_datasets()
        self.attack_model       = load_model_defination(attack_model_name, num_channels = 1, num_classes = 2).to(device)
        self.shadow_epochs      = shadow_epochs
        self.attack_epochs      = attack_epochs
        self.wandb_logging      = wandb_logging
        self.shadow_models_trained = False

        
        
        

    def get_shadow_datasets(self):
        initial_dataset = self.build_master_shadow_dataset(self.target_trainset, self.target_testset)
        return self.get_final_shadow_datasets(initial_dataset)


    def build_master_shadow_dataset(self, trainset, testset):
        
        trainset_size = len(trainset)
        seen_size = int(trainset_size * self.seen_fraction) 
        seen_dataset =  Subset(trainset, torch.randperm(trainset_size)[:seen_size])
        
        testset_size = len(testset)
        unseen_size = int(testset_size * self.unseen_fraction)
        unseen_dataset =  Subset(testset, torch.randperm(testset_size)[:unseen_size])

        print(f"Seen size: {seen_size}, unseen size: {unseen_size}")    

        return ConcatDataset([seen_dataset, unseen_dataset])   

        # print(f"Initial dataset ballance: {build_label_dict(initial_dataset)}")

        # return initial_dataset

    

    def get_final_shadow_datasets(self, initial_dataset ):
        # Split dataset into `self.shadow_count` overlapping partitions, with 'overlap_fraction' overlap, to simulate different shadow datasets
        total_size = len(initial_dataset)
        partition_size = int(total_size*self.overlap_fraction) 

        
        

        datasets = []
        # dataset_ballance_dicts = []
        for i in range(self.shadow_count):
            datasets.append( Subset(initial_dataset, torch.randperm(total_size)[:partition_size]) )
        #     ds = datasets[i]
        #     bal_dict = build_label_dict(ds)

        #     dataset_ballance_dicts.append(bal_dict)
            

            

        # pdb.set_trace()

        # Split each partition into train/val/test and create DataLoader
        trainloaders = []
        valloaders = []
        testloaders = []
        for ds in datasets:
            len_test    = int(len(ds) * self.shadow_test_fraction)      # test set size
            len_train = len(ds) - len_test      # train set size
            lengths = [len_train, len_test]
            # assert len_test +  len_train == len(ds)
            ds_train, ds_test = random_split(ds, lengths, torch.Generator().manual_seed(42))

        

            try:
                trainloaders.append(DataLoader(ds_train, self.batch_size, shuffle=True))
                testloaders.append(DataLoader(ds_test, self.batch_size))
            except Exception as e:
                traceback.print_exc()
                pdb.set_trace()
                
        return trainloaders, testloaders
    
    

    


    def build_attack_dataset(self):
        if not self.shadow_models_trained:
            raise Exception("Shadow models not trained")
        partial_dataset = []
        for i in range(self.shadow_count):
            shadow_model    = self.shadow_models[i]
            print(f'train size: {len(self.shadow_train_dataloader[i])}, test size: {len(self.shadow_test_dataloader[i])}')
            shadow_dataset  = [self.shadow_train_dataloader[i], self.shadow_test_dataloader[i]]
            partial_dataset.append(  Loss_Label_Dataset(shadow_dataset, shadow_model) )

        loss_dataset = ConcatDataset(partial_dataset)
        loss_dataset_size = len(loss_dataset)

        
        
        val_size    = int(loss_dataset_size * self.val_fraction)
        train_size  = loss_dataset_size - val_size

        save_loss_dataset(loss_dataset, f'loss_dataset_class_{self.class_id}')

        attack_trainset, attack_valset = random_split(loss_dataset, [train_size, val_size])
        
        attack_trainloder       = DataLoader(attack_trainset, self.batch_size, shuffle=True)
        attack_valloder         = DataLoader(attack_valset, self.batch_size)

        target_dataset= load_dataloaders(self.target_trainset, self.target_testset)

        attack_testloder        = DataLoader(Loss_Label_Dataset(target_dataset, self.target_model), self.batch_size)
        # pdb.set_trace()

        return attack_trainloder, attack_valloder, attack_testloder


    def run_membership_inference_attack(self):
        
        # runs membership inference attack on the target model
        # creates a training loop where the output of the shadow model is trained to mimic the target model
        # self.target_trainset, self.target_testset = load_dataloaders(self.target_trainset, self.target_testset)
        target_dataloader=  DataLoader(self.target_testset, self.batch_size) 
        loss, accuracy = test(self.target_model, target_dataloader)
        print(f'\nFor the target model on the target test set, Loss: {loss}, Accuracy: {accuracy}')
        for i in range(self.shadow_count):
            print_info(self.device, model_name=f'shadow model {i}', dataset_name=f'shadow dataset {i}', teacher_name=self.target_model.__class__.__name__)
            train_shadow_model(self.target_model, 
                               self.shadow_models[i], 
                               self.shadow_train_dataloader[i], 
                               self.shadow_test_dataloader[i], 
                               self.shadow_epochs, 
                               verbose=False, 
                               wandb_logging=self.wandb_logging
                               )
            loss, accuracy = test(self.shadow_models[i], target_dataloader)
            print(f'\nFor the shadow model {i} on the target test set, Loss: {loss}, Accuracy: {accuracy}')
            
        

        self.shadow_models_trained = True
        attack_trainloder, attack_valloder, attack_testloder = self.build_attack_dataset()

        
            
        
        print_info(self.device, model_name='attack model', dataset_name=f'attack dataset')
        train(  self.attack_model, 
                attack_trainloder, 
                attack_valloder, 
                epochs=self.attack_epochs,
                device=self.device,
                verbose=False, 
                wandb_logging =self.wandb_logging,
                is_binary=True
                )
        
        loss, accurecy = test(self.attack_model, attack_testloder, device=self.device, is_binary=True)

        # loss, accurecy = test(self.attack_model, attack_testloder, device=self.device, binary=True)

        return loss, accurecy
    
   
def main():
    parser = argparse.ArgumentParser(description='A description of your program')
    parser.add_argument('-w', '--wandb_logging', action='store_true', help='Enable wandb logging')
    parser.add_argument('-e', '--num_shadow_epochs', type=int, default=50, help='Number of rounds of shadow training')
    parser.add_argument('-e1', '--num_attack_epochs', type=int, default=50, help='Number of rounds of attack training')
    parser.add_argument('-d', '--dataset_name', type=str, default='CIFAR10', help='Dataset name, target dataset ')
    parser.add_argument('-n', '--shadow_count', type=int, default=8, help='Number of shadow models')
    parser.add_argument('-m', '--target_model_name', type=str, default='basicCNN', help='Model name for the model to be attacked')
    parser.add_argument('-mw', '--target_model_weights', type=str, default='basicCNN', help='Weights for the model to be attacked')
    parser.add_argument('-s', '--shadow_model_name', type=str, default='basicCNN', help='Model name for the shadow model')
    parser.add_argument('-a', '--attack_model_name', type=str, default= 'attack_classifier', help='Classifier for the attack model')
    args = parser.parse_args()


    device = get_device()
    num_channels, num_classes = get_datasets_details(args.dataset_name)

    target_model        = load_model_defination(args.target_model_name, num_channels, num_classes).to(device)
    load_saved_weights(target_model, filename =args.target_model_weights)

    attack = Membership_inference_attack(target_model, args.dataset_name, 
                                         args.shadow_model_name, 
                                         args.shadow_count,
                                         args.attack_model_name, 
                                         device, 
                                         args.num_shadow_epochs, args.num_attack_epochs,
                                         args.wandb_logging
                                         )

    attack.start()




if __name__ == '__main__':
    main()