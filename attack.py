import argparse, os
import pdb 
import traceback
import torch
import wandb
from copy import deepcopy
from torch.utils.data import  DataLoader, ConcatDataset, Subset, random_split 
from utils.models import load_model_defination
from utils.training_utils import get_device, save_loss_dataset, load_loss_dataset, train_shadow_model, wandb_init, print_info, save_model,train, test, load_model as load_saved_weights
from utils.datasets import DatasetWrapper, ContinuousDatasetWraper, Loss_Label_Dataset, load_dataloaders


class Classwise_membership_inference_attack:
    def __init__(self, target_model, target_dataset, attack_instance, target_model_name, wandb_logging=False):
        
        # self.shadow_count = shadow_count

        self.target_model           = target_model
        self.target_model_name      = target_model_name
        # self.target_dataset_name    = target_dataset_name
        # _, self.num_classes         = get_datasets_details(self.target_dataset_name)
        self.target_dataset         = target_dataset
        # self.datasets_details       = get_datasets_details(target_dataset_name)
        self.attack_instance        = attack_instance
        self.wandb_logging          = wandb_logging    

    def split_dataset_by_label(self, dataset):   
        class_datasets = [[] for _ in range(self.target_dataset.num_classes)]      

        # Assign samples to their respective class lists
        for sample, label in dataset:
            class_datasets[label].append((sample, label))

        return class_datasets
    
    def start(self):
        
        class_wise_trainset    = self.split_dataset_by_label(self.target_dataset.trainset)
        class_wise_testset     = self.split_dataset_by_label(self.target_dataset.testset)
        loss, accuracy = 0.0, 0.0
        predictions = [[],[]]
        for c in range(self.target_dataset.num_classes):
            print(f'\nMembership Inference Attack Instance on class {c} initialized')
            class_wise_datasets = [class_wise_trainset[c], class_wise_testset[c]]
            attack_instance = deepcopy(self.attack_instance)

            attack_instance.define_target_model_and_datasets(c, self.target_model, class_wise_datasets, self.target_dataset, self.target_model_name)
            class_loss, class_accuracy, class_predictions = attack_instance.run_membership_inference_attack()
            try:
                predictions[0].extend(class_predictions[0])
                predictions[1].extend(class_predictions[1])
            except:
                traceback.print_exc()
                import pdb
                pdb.set_trace()
            print(f"\n\tLoss on class {c}: {class_loss}, and Accuracy on class {c}: {class_accuracy}")
            loss        += class_loss
            accuracy    += class_accuracy
            del attack_instance
        
        loss     /= self.target_dataset.num_classes
        accuracy /= self.target_dataset.num_classes

        print(f"\nOverall Loss: {loss}, and Overall Accuracy: {accuracy}")
        
        if self.wandb_logging:
            # input("Press Enter to Log run with wandb...")
            wandb_init(dataset_name= self.target_dataset.name, 
                       model_name = self.target_model.__class__.__name__, 
                       comment = f"classwise_membership_inference_attack_overall_{self.target_model_name}"
                       )
            wandb.log({"test_acc": accuracy, "test_loss": loss})
            wandb.finish()
        return loss, accuracy, predictions
        

class Combined_membership_inference_attack(Classwise_membership_inference_attack):
    def __init__(self, target_model, target_dataset, attack_instance, target_model_name, wandb_logging=False):
        super().__init__(target_model, target_dataset, attack_instance, target_model_name, wandb_logging)

    def start(self):
        loss, accuracy = 0.0, 0.0
        
        print(f'\nMembership Inference Attack Instance on all classes at once initialized')

        
        class_wise_datasets = [self.target_dataset.trainset, self.target_dataset.testset]

        self.attack_instance.define_target_model_and_datasets(-1, self.target_model, class_wise_datasets, self.target_dataset, self.target_model_name)
        loss, accuracy, predictions = self.attack_instance.run_membership_inference_attack()
        
        # del self.attack_instance
        
        # loss     /= self.num_classes
        # accuracy /= self.num_classes

        print(f"\nOverall Loss: {loss}, and Overall Accuracy: {accuracy}")
        # pdb.set_trace()
        
        # if self.wandb_logging:
        #     # input("Press Enter to Log run with wandb...")
        #     wandb_init(dataset_name= self.target_dataset_name, 
        #                model_name = self.target_model.__class__.__name__, 
        #                comment = f"classwise_membership_inference_attack_overall_{self.target_dataset_name}_{self.target_model.__class__.__name__}"
        #                )
        #     wandb.log({"test_acc": accuracy, "test_loss": loss})
        #     wandb.finish()
        return loss, accuracy, predictions

        



class Membership_inference_attack_instance:
    def __init__(self, 
                 shadow_model_name, 
                 shadow_count, 
                 load_attack_dataset,
                 save_attack_dataset,
                 save_attack_model,
                 attack_model_name, 
                 batchwise_loss,
                 device, 
                 shadow_distilled=False,
                 shadow_epochs=50, 
                 attack_epochs=50, 
                 wandb_logging=False
                 ):

        
        self.overlap_fraction           = 0.8
        self.seen_fraction              = 0.12 
        self.unseen_fraction            = 0.6
        self.val_fraction               = 0.2
        self.shadow_test_fraction       = 0.5
        self.batch_size                 = 32
        self.device                     = device
        self.wandb_logging              = wandb_logging
        self.save_attack_model          = save_attack_model
        self.batchwise_loss             = batchwise_loss
        self.save_attack_dataset        = save_attack_dataset
        self.load_saved_attack_dataset  = load_attack_dataset
        try:
            assert self.save_attack_dataset != self.load_saved_attack_dataset
        except AssertionError as e:
            self.load_saved_attack_dataset = not self.save_attack_dataset
            # raise AssertionError(f'{self.save_attack_dataset=} and {self.load_saved_attack_dataset=} cant both be same')
            
            # pdb.set_trace()
        self.shadow_epochs              = shadow_epochs
        self.attack_epochs              = attack_epochs
        self.shadow_models_trained      = False
        self.shadow_distilled           = shadow_distilled
        self.attack_dataset_built       = False
        self.target_defined             = False 
        self.shadow_count               = shadow_count        
        self.shadow_models              = []
        self.shadow_model_name          = shadow_model_name                
        self.attack_model               = load_model_defination(attack_model_name, num_channels = 1, num_classes = 2).to(device)

    def __del__(self):
        """
        If wandb_logging is enabled, closes the wandb session
        """
        if self.wandb_logging:
            wandb.finish()
    
    def define_target_model_and_datasets(self, class_id, target_model, class_wise_datasets, target_dataset, target_model_name):       
        
        self.class_id                   = class_id
        self.target_model               = target_model 
        self.target_dataset             = target_dataset
        self.target_model_name          = target_model_name 
        self.target_dataset.trainset    = class_wise_datasets[0] 
        self.target_dataset.testset     = class_wise_datasets[1]
        self.target_defined             = True


        for _ in range(self.shadow_count):
            self.shadow_models.append(load_model_defination(self.shadow_model_name, self.target_dataset.num_channels, self.target_dataset.num_classes).to(self.device))

        self.shadow_train_dataloader, self.shadow_test_dataloader    = self.get_shadow_datasets()
        if self.wandb_logging:
            classID = 'all_at_once' if self.class_id == -1 else self.class_id

            
            wandb_init(model_name = self.attack_model.__class__.__name__, 
                       dataset_name = 'Loss_Label_Dataset', 
                       comment = f'MIA_{self.target_model_name}_attack_class_{classID}')
    
    def get_shadow_datasets(self):
        if not self.target_defined:
            raise Exception("Target model and datasets not defined yet")
        initial_dataset = self.build_master_shadow_dataset(self.target_dataset.trainset, self.target_dataset.testset)
        return self.get_final_shadow_datasets(initial_dataset)

    def build_master_shadow_dataset(self, trainset, testset):
        
        trainset_size = len(trainset)
        seen_size = int(trainset_size * self.seen_fraction) 
        seen_dataset =  Subset(trainset, torch.randperm(trainset_size)[:seen_size]) # type: ignore
        
        testset_size = len(testset)
        unseen_size = int(testset_size * self.unseen_fraction)
        unseen_dataset =  Subset(testset, torch.randperm(testset_size)[:unseen_size]) # type: ignore

        # print(f"Seen size: {seen_size}, unseen size: {unseen_size}")    

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
            datasets.append( Subset(initial_dataset, torch.randperm(total_size)[:partition_size]) ) # type: ignore
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
                # pdb.set_trace()
                
        return trainloaders, testloaders


    def build_attack_loaders(self, loss_dataset):
        loss_dataset_size = len(loss_dataset)      
        
        val_size    = int(loss_dataset_size * self.val_fraction)
        train_size  = loss_dataset_size - val_size

        attack_trainset, attack_valset = random_split(loss_dataset, [train_size, val_size])
        
        self.attack_trainloder       = DataLoader(attack_trainset, self.batch_size, shuffle=True)
        self.attack_valloder         = DataLoader(attack_valset, self.batch_size)

        target_dataset= load_dataloaders(self.target_dataset.trainset, self.target_dataset.testset)

        self.attack_testloder        = DataLoader(Loss_Label_Dataset(target_dataset, self.target_model, self.device, loss_batchwise=False), self.batch_size)

        # pdb.set_trace()
        
        
        self.attack_dataset_built   = True 

    def build_attack_dataset(self):
        """
        Builds the attack dataset.

        If the shadow models have not been trained, it initializes training of the shadow models.
        Builds the attack dataset using the shadow models.

        Args:
            self (object): The instance of the class.

        Returns:
            None
        """
        if not self.shadow_models_trained:
            # raise Exception("Shadow models not trained")
            self.train_shadow_model()

        partial_dataset = []
        inverse_loss_mode = not self.batchwise_loss # for saving the loss dataset not defined by self.batchwise_loss        
        partial_dataset2 = []  #  (both are saved at once) 
        for i in range(self.shadow_count):
            shadow_model    = self.shadow_models[i]
            # print(f'train size: {len(self.shadow_train_dataloader[i])}, test size: {len(self.shadow_test_dataloader[i])}')
            shadow_dataset  = [self.shadow_train_dataloader[i], self.shadow_test_dataloader[i]]
            partial_dataset.append(  Loss_Label_Dataset(shadow_dataset, shadow_model, self.device, loss_batchwise=self.batchwise_loss) )
            partial_dataset2.append(  Loss_Label_Dataset(shadow_dataset, shadow_model, self.device, loss_batchwise= inverse_loss_mode) )


        loss_dataset = ConcatDataset(partial_dataset)
        other_loss_dataset = ConcatDataset(partial_dataset2)

        if self.save_attack_dataset:
            (suffix1, suffix2) = ('batch', 'single') if self.batchwise_loss else ('single', 'batch')
            self.save_loss_dataset_with_suffix(loss_dataset, suffix1)
            self.save_loss_dataset_with_suffix(other_loss_dataset, suffix2)
            for batch_size in [8, 16, 64, 128, 256]:
                self.other_batchwise_loss(batch_size)

            


        self.build_attack_loaders(loss_dataset)


    def other_batchwise_loss(self, batch_size = 32):
        partial_dataset = []
        for i in range(self.shadow_count):
            shadow_model    = self.shadow_models[i]
            # print(f'train size: {len(self.shadow_train_dataloader[i])}, test size: {len(self.shadow_test_dataloader[i])}')
            shadow_dataset  = [self.shadow_train_dataloader[i], self.shadow_test_dataloader[i]]
            partial_dataset.append(  Loss_Label_Dataset(shadow_dataset, shadow_model, self.device, loss_batchwise= True, batch_size = batch_size) )
        loss_dataset = ConcatDataset(partial_dataset)
        self.save_loss_dataset_with_suffix(loss_dataset, f'batch_{batch_size}')

    def save_loss_dataset_with_suffix(self, loss_dataset, suffix):
        directory = f'{self.target_model_name}_{suffix}'
        file_path = os.path.join(directory, 'loss_dataset_class_' + str(self.class_id))
        save_loss_dataset(loss_dataset, file_path)

    def load_attack_dataset(self):
        """
        Loads the attack dataset for the given class.

        This function attempts to load the saved attack dataset for the specified class ID. 
        If the dataset is found, it is used to build the attack loaders. 
        If the dataset is not found, this function builds the attack dataset and then the attack loaders.

        Parameters:
            self (object): The current instance of the class.

        Returns:
            None
        """
        try:
            classID = 'combined class' if self.class_id == -1 else self.class_id
            print(f'\tLoading saved attack dataset for class {classID}')
            if type(self.batchwise_loss) == bool:
                suffix= 'batch' if self.batchwise_loss else 'single'
            else:
                suffix = self.batchwise_loss
            directory = f'{self.target_model_name}_{suffix}'
            file_path = os.path.join(directory, 'loss_dataset_class_' + str(self.class_id))
            loss_dataset = load_loss_dataset(file_path)
            self.build_attack_loaders(loss_dataset)
        except FileNotFoundError as e:
            traceback.print_exc()
            print(f'{file_path} not found') # type: ignore
            # pdb.set_trace()
            # print(f'\tAttack dataset for class {classID} not found, building dataset...') # type: ignore
            # self.build_attack_dataset()      
            # raise e  
   
    def train_shadow_model(self):
        target_dataloader=  DataLoader(self.target_dataset.testset, self.batch_size) 
        loss, accuracy, _ = test(self.target_model, target_dataloader)
        print(f'\n\tFor the target model on the target test set, Loss: {loss}, Accuracy: {accuracy}')

        for i in range(self.shadow_count):
            if self.shadow_distilled:    # distillation the knoledge from the target model to the shadow model
                print_info(self.device, model_name=f'shadow model {i}', dataset_name=f'shadow dataset {i}', teacher_name=self.target_model.__class__.__name__)
                train_shadow_model(self.target_model, 
                                self.shadow_models[i], 
                                self.shadow_train_dataloader[i], 
                                self.shadow_test_dataloader[i], 
                                self.shadow_epochs, 
                                verbose=False, 
                                wandb_logging=self.wandb_logging
                                )
            else:  # Train the shadow model outright on the corresponding shadow dataset
                print_info(self.device, model_name=f'shadow model {i}', dataset_name=f'shadow dataset {i}')
                train(self.shadow_models[i], 
                    self.shadow_train_dataloader[i], 
                    self.shadow_test_dataloader[i], 
                    self.shadow_epochs, 
                    verbose=False, 
                    wandb_logging=self.wandb_logging
                    )
            loss, accuracy, _ = test(self.shadow_models[i], target_dataloader)
            print(f'\n\tFor the shadow model {i} on the target test set, Loss: {loss}, Accuracy: {accuracy}')
        self.shadow_models_trained = True

    def train_attack_model(self):
        if not self.attack_dataset_built:
            raise  Exception("Attack dataset not built")
        classID = 'combined' if self.class_id == -1 else self.class_id
        print_info(self.device, model_name=f'attack model {classID}', dataset_name=f'attack dataset', no_FL=True)
        train(  self.attack_model, 
                self.attack_trainloder, 
                self.attack_valloder, 
                epochs=self.attack_epochs,
                device=self.device,
                verbose=False, 
                wandb_logging =self.wandb_logging,
                is_binary=True
                )
        
    def run_membership_inference_attack(self):
        """
        Runs the membership inference attack.

        If the saved attack dataset is available, trains the shadow model, and builds the attack dataset.
        Otherwise, loads the attack dataset.

        Trains the attack model.

        Calculates the loss and accuracy of the attack model using the attack test loader.

        Returns:
            - loss (float): The loss of the attack model during testing.
            - accuracy (float): The accuracy of the attack model during testing.
        """
        
        if not self.load_saved_attack_dataset:
            self.train_shadow_model()         
            self.build_attack_dataset()
        else:
            self.load_attack_dataset()

        self.train_attack_model()

        # plot_roc = self.load_saved_attack_dataset
        
        # pdb.set_trace()
            
        loss, accuracy, predictions = test(self.attack_model, self.attack_testloder, device=self.device, is_binary=True, plot_ROC=False)
        # loss, accuracy, predictions = 0,0, [[0,1],[0,1]]
        # pdb.set_trace()


        if self.wandb_logging:
            wandb.log({"test_acc": accuracy, "test_loss": loss})
            


        if self.save_attack_model:
            save_model(self.attack_model ,filename =f'Attack_model_{self.class_id}', print_info=False)

        return loss, accuracy, predictions
        

def argument_parser():
    parser = argparse.ArgumentParser(description='A description of your program')
    parser.add_argument('-w', '--wandb_logging', action='store_true', help='Enable wandb logging')
    
    parser.add_argument('-sm', '--save_attack_model', action='store_true', help='save the developed attack model')
    parser.add_argument('-e', '--num_shadow_epochs', type=int, default=50, help='Number of rounds of shadow training')
    parser.add_argument('-b', '--batchwise_loss', action='store_true', help='For attack model training, loss is calculated batchwise, otherwise loss is calculated samplewise')
    parser.add_argument('-c', '--combined_class', action='store_true', help='if tihs flag is present, combined class attack, otherwise classwise separate attack')
    parser.add_argument('-e1', '--num_attack_epochs', type=int, default=50, help='Number of rounds of attack training')
    parser.add_argument('-d', '--dataset_name', type=str, default='CIFAR10', help='Dataset name, target dataset ')
    parser.add_argument('-n', '--shadow_count', type=int, default=8, help='Number of shadow models')
    parser.add_argument('-m', '--target_model_name', type=str, default='basicCNN', help='Model name for the model to be attacked')
    parser.add_argument('-mw', '--target_model_weights', type=str, default='centralizedbasicCNN', help='Weights for the model to be attacked')
    parser.add_argument('-s', '--shadow_model_name', type=str, default='basicCNN', help='Model name for the shadow model')
    parser.add_argument('-a', '--attack_model_name', type=str, default= 'attack_classifier', help='Classifier for the attack model')
    parser.add_argument('-ds', '--distil_shadow_model', action='store_true', help='For shadow model training, The shadow models are is distilled from the target model, otherwise the shadow models are trained from scratch')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-ld', '--load_attack_dataset', action='store_true', help='Instead of building attack dataset, load pre-existing attack dataset from disc')
    group.add_argument('-sv', '--save_attack_dataset', action='store_true', help='Save computed attack dataset to disc')

    parser.set_defaults(load_attack_dataset=True)
    args = parser.parse_args()   
    return args    
    
   
def main(args): 
    device = get_device()
    # pdb.set_trace()
    try:
        target_dataset = DatasetWrapper(args.dataset_name)
    except NotImplementedError as e:
        # dataset_name_and_index = 
        dataset_name, index = args.dataset_name.split('-')
        print(f'Loading {dataset_name} with index {index}')
        target_dataset = ContinuousDatasetWraper(dataset_name)
        target_dataset.select_split(int(index))
    
    mode    = 'Combined Class' if args.combined_class else 'Classwise'
    suffix  = 'batchwise' if args.batchwise_loss else 'samplewise'

    print(f'\n\nExecuting {mode} Membership Inference Attack on {args.target_model_weights}, computing loss for training attack model {suffix}\n\n')

    target_model        = load_model_defination(args.target_model_name, target_dataset.num_channels, target_dataset.num_classes).to(device)
    load_saved_weights(target_model, filename =args.target_model_weights)
    attack_instance = Membership_inference_attack_instance( shadow_model_name   = args.shadow_model_name, 
                                                            shadow_count        = args.shadow_count, 
                                                            load_attack_dataset = args.load_attack_dataset,
                                                            save_attack_dataset = args.save_attack_dataset,
                                                            save_attack_model   = args.save_attack_model,
                                                            attack_model_name   = args.attack_model_name,
                                                            batchwise_loss      = args.batchwise_loss,
                                                            device              = device,
                                                            shadow_distilled    = args.distil_shadow_model,
                                                            shadow_epochs       = args.num_shadow_epochs,
                                                            attack_epochs       = args.num_attack_epochs,
                                                            wandb_logging       = args.wandb_logging
                                                        )

    if args.combined_class:
        attack = Combined_membership_inference_attack(target_model, target_dataset, attack_instance, args.target_model_weights, args.wandb_logging)
    else:
        attack = Classwise_membership_inference_attack(target_model, target_dataset, attack_instance, args.target_model_weights, args.wandb_logging)
    
    


    attack.start()




if __name__ == '__main__':
    args = argument_parser()
    main(args)