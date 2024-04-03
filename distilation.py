import argparse

import wandb
from utils.datasets import load_partitioned_dataloaders
from utils.models import load_model_defination
from utils.training_utils import get_device, print_info, save_model, test, train_shadow_model, load_model as load_saved_weights, wandb_init


class Model_Distilation:
    def __init__(self, teacher_model_name, teacher_weights, student_model_name, student_weights, dataset_name='CIFAR10', device = get_device(),  wandb_logging=False):
        self.teacher_model_name                         = teacher_model_name
        self.teacher_weights                            = teacher_weights
        self.student_weights                            = student_weights
        self.student_model_name                         = student_model_name
        self.dataset_name                               = dataset_name                     
        loaders,  self.num_channels, self.num_classes   = load_partitioned_dataloaders(num_clients= 1, dataset_name = self.dataset_name, val_percent = 10, batch_size=32)
        [trainloaders, valloaders, self.testloader, _]  = loaders
        self.trainloader                                = trainloaders[0]
        self.valloader                                  = valloaders[0]  
        self.device                                     = device
        self.wandb_logging                              = wandb_logging

    def distil_for_epochs(self, epochs, optimizer = None, criterion = None,  verbose=False, patience= 5):   
        print_info(self.device, self.student_model_name, self.dataset_name, self.teacher_weights)
                
        teacher = load_model_defination(self.teacher_model_name, self.num_channels, self.num_classes).to(self.device)
        try:
            load_saved_weights(teacher, filename =self.teacher_weights)
        except FileNotFoundError:
            print("Teacher weights not found. Fatal error")   
            raise SystemError  

        student = load_model_defination(self.student_model_name, self.num_channels, self.num_classes).to(self.device)
        try:
            load_saved_weights(student, filename =self.student_weights)
        except FileNotFoundError:
            print("Student weights not found. Proceding with distillation") 

        pre_loss, pre_accuracy, _ = test(student, self.testloader)
        print(f"\n\nPre-Distillation test set performance:\n\tloss {pre_loss}\n\taccuracy {pre_accuracy}\n\n")

        train_shadow_model(teacher, student, self.trainloader, self.valloader, 
                           epochs, optimizer, criterion, self.device, 
                           wandb_logging = self.wandb_logging, accuracy_defined=True,
                           patience=patience, verbose=False)
        
        post_loss, post_accuracy, _ = test(student, self.testloader)
        print(f"\n\nPost-Distillation test set performance:\n\tloss {post_loss}\n\taccuracy {post_accuracy}\n\n")
        return student

def main():
    parser = argparse.ArgumentParser(description='A description of your program')
    parser.add_argument('-e', '--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-d', '--dataset_name', type=str, default='CIFAR10', help='Dataset name')
    parser.add_argument('-s', '--student_model_name', type=str, default='basicCNN', help='Name of model to be distilled')
    parser.add_argument('-sw','--student_weights', type=str, default= 'CentralizedbasicCNNCIFAR10', help='filename of the pretrained weights for the student model')
    parser.add_argument('-t', '--teacher_model_name', type=str, default= 'efficientnet', help='Name of the pretrained model')
    parser.add_argument('-tw','--teacher_weights', type=str, default= 'EfficientNet', help='filename of the pretrained weights for the teacher model')
    parser.add_argument('-w', '--wandb_logging', action='store_true', help='Enable wandb logging') 
    parser.add_argument('-o', '--overfit_patience', type=int, default=5, help='Patience after which to stop training, to prevent overfitting')
    parser.add_argument('-c', '--comment', type=str, default='', help='Comment for this run')
    args = parser.parse_args()

    device = get_device()
    if args.comment == '':
        args.comment = 'Distilation_'+args.teacher_model_name+'>'+args.student_model_name+'_'+args.dataset_name

    if args.wandb_logging:
        wandb_init(comment=args.comment, model_name=args.teacher_model_name+'>'+args.student_model_name, dataset_name=args.dataset_name)
        # wandb.watch(model, log_freq=100)
    
    distiler = Model_Distilation(args.teacher_model_name, args.teacher_weights, args.student_model_name, args.student_weights, args.dataset_name, device, args.wandb_logging)

    trained_student_model = distiler.distil_for_epochs(args.num_epochs)
    filename =  args.student_model_name+'_distiled_with_'+args.teacher_model_name
    save_model(trained_student_model,filename=filename, print_info=True)

    if args.wandb_logging:
        wandb.finish()

if __name__ == '__main__':
    main()