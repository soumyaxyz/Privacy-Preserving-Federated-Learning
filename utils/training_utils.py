from collections import OrderedDict
from typing import Dict, List
import numpy as np
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import pdb,traceback
import os
import csv
import json

from utils.datasets import Wrapper_Dataset
from utils.plot_utils import plot_ROC_curve
from utils.lib import blockPrintingIfServer, create_directories_if_not_exist

def wandb_init(
    project="Privacy_Preserving_Federated_Learning", 
    entity='', 
    model_name="basicCNN", 
    dataset_name="CIFAR_10",
    comment= '',  
    lr ='', 
    optimizer = ''
    ):
    config_path = os.path.join('wandb', 'config.json')
    with open(config_path) as config_file:
        config = json.load(config_file)  
        api_key = config.get('api_key') 
        if entity == '':
            entity = config.get('entity')
         
    wandb.login( key=api_key )
    wandb.init(
      project=project, entity=entity,
      config={"learning_rate": lr, "optimiser": optimizer, "comment" : comment, "model": model_name, "dataset": dataset_name}
    )


def get_device():
    if torch.cuda.is_available():
      return torch.device("cuda")
    else:
      return torch.device("cpu")

def print_info(device, model_name="model", dataset_name="dataset", teacher_name=None, no_FL = False):
    if device.type=="cuda":
        device_type = torch.cuda.get_device_name(0)  
    else:
        device_type = device.type
        
    if teacher_name:
        print(f"\nDistiling {model_name} from {teacher_name} on {dataset_name} in {device_type} using PyTorch {torch.__version__}")
    elif no_FL:
        print(f"\n\tTraining {model_name} with {dataset_name} in {device_type} using PyTorch {torch.__version__}")
    else:
        print(f"\nTraining {model_name} with {dataset_name} in {device_type} using PyTorch {torch.__version__} and Flower {fl.__version__}")

def verify_folder_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")
    return path

def save_model(net, optim = None, filename ='filename', print_info=False):
    sanatized_filename = "".join(x for x in filename if x.isalnum())
    save_folder = './saved_models/'
    path = verify_folder_exist(save_folder)+sanatized_filename+'.pt'
    if optim:
        torch.save({'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optim.state_dict()
            }, path)
    else:
        torch.save({'model_state_dict': net.state_dict()}, path)

    if print_info:
        print(f"\nSaved model to {path}")

def load_model(net, optim=None, filename ='filename', print_info=False):
    sanatized_filename = "".join(x for x in filename if x.isalnum())
    path = './saved_models/'+sanatized_filename+'.pt'
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])
    if optim:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
    if print_info:
        print(f"Loaded model from {path}")

def delete_saved_model(filename ='filename', print_info=False):
    sanatized_filename = "".join(x for x in filename if x.isalnum())
    path = './saved_models/'+sanatized_filename+'.pt'
    if os.path.exists(path):
        os.remove(path)
    if print_info:
        print(f"Deleted model from {path}")


def save_loss_dataset(dataset, filename='datset'):    
    print(f'\tSaving dataset of size {len(dataset)} to {filename}')
    save_path = './saved_models/'+filename+'.csv'
    create_directories_if_not_exist(save_path)
    with open(save_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['data', 'label'])  # Write the header row

        for data, label in dataset:
            # Convert tensors to numpy arrays if needed
            data = data.numpy() if isinstance(data, torch.Tensor) else data
            label = label.numpy() if isinstance(label, torch.Tensor) else label

            writer.writerow([data, label])  # Write each data and label as a row





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


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def loss_fn_kd(outputs, labels, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = params.alpha
    T = params.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


@blockPrintingIfServer
def train_single_epoch(net, trainloader, optimizer = None, criterion = None, device = get_device(), is_binary=False):
    """Train the network on the training set."""
    if not criterion:
        criterion = torch.nn.CrossEntropyLoss()
    if not optimizer:
        optimizer = torch.optim.Adam(net.parameters())
    net.train()
    correct, total, epoch_loss = 0, 0, 0.0
    for images, labels in trainloader:
        try:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss.item()
            total += labels.size(0)
            if is_binary:
                correct += (torch.round(outputs.data) == labels).sum().item()
            else:
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        except Exception as e:
            traceback.print_exc()
            pdb.set_trace()
    epoch_loss /= len(trainloader.dataset)
    epoch_acc = correct / total
    # print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
    return epoch_loss, epoch_acc

@blockPrintingIfServer
def test(net, testloader, device = get_device(), is_binary=False, plot_ROC=False):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    predictions = None
    try:
        # if plot_ROC:
        gold = []
        pred = []
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss += criterion(outputs, labels).item()                
                total += labels.size(0)
                if is_binary:
                    correct += (torch.round(outputs.data) == labels).sum().item()
                else:
                    correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

                if plot_ROC:
                    gold = np.append(gold, labels.cpu().numpy()) # type: ignore
                    if is_binary:
                        pred = np.append(pred, torch.round(outputs).cpu().numpy())# type: ignore
                    else:
                        pred = np.append(pred, torch.max(outputs, 1)[1])  # unverified # type: ignore
                else:
                    # pdb.set_trace()
                    ( confidence, prediction) = torch.max(outputs, 1)
                    confidence = confidence.cpu().numpy()
                    prediction = prediction.cpu().numpy()
                    truth =labels.cpu().numpy()
                    result = (prediction == truth).astype(np.int64)
                    pred = np.append(pred, confidence) #prediction comfidece
                    # pdb.set_trace()

                    gold = np.append(pred, result)

                    # try:
                    #     assert len(confidence) == len(gold)
                    # except AssertionError:
                    #     traceback.print_exc()
                    #     pdb.set_trace()
                

        loss /= len(testloader.dataset)
        accuracy = correct / total
        predictions = [pred, gold]  # type: ignore   
        if plot_ROC:            
            plot_ROC_curve(gold, pred)  # type: ignore          
            # pdb.set_trace()
            
    except Exception as e:
        traceback.print_exc()
        pdb.set_trace()
        
    return loss, accuracy, predictions # type: ignore

@blockPrintingIfServer
def train(net, trainloader, valloader, epochs: int, optimizer = None, criterion = None, device=get_device(), verbose=False, wandb_logging=True, patience= 5, loss_min = 100000, is_binary=False):
    """Train the network on the training set."""
    if not criterion:
        criterion = torch.nn.CrossEntropyLoss()
    if not optimizer:
        optimizer = torch.optim.Adam(net.parameters())

    savefilename = net.__class__.__name__

    record_mode = False

    initial_patience = patience

    if not verbose:
        pbar = tqdm(total=epochs, position=1, leave=False)
        pbar2 = tqdm(total=patience, position=2, leave=False)
        pbar2.update(patience)
        pbar.set_description(f"Epoch {1}")
        pbar2.set_description(f"patience")
    for epoch in range(epochs):
        if record_mode:
            if wandb_logging:
                wandb.log({"acc": accuracy,"loss": loss}) # type: ignore
            if not verbose:
                pbar.update(1) # type: ignore
        elif patience<= 0:
            try:
                load_model(net, optimizer, savefilename)
            except Exception as e:
                print(traceback.print_exc())
                pdb.set_trace()
            loss, accuracy, _ = test(net, valloader, device, is_binary)
            if wandb_logging:
                wandb.log({"acc": accuracy,"loss": loss}) 
            if not verbose:
                pbar.update(1)  # type: ignore
                pbar2.set_description(f"Early stopped at epoch {epoch+1}, train_loss: {train_loss:.4f}, loss: {loss:.4f}, train_acc {train_acc:.4f}, acc: {accuracy:.4f}") # type: ignore
            # break
            record_mode = True
        else:
            train_loss, train_acc =  train_single_epoch(net, trainloader, optimizer, criterion, device, is_binary) 
            loss, accuracy, _ = test(net, valloader, device, is_binary)

            if loss_min > loss: # validation loss improved
                patience = initial_patience
                loss_min = loss
                save_model(net, optimizer, savefilename)
            else:
                patience -= 1

            if wandb_logging:
                wandb.log({"train_acc": train_acc, "train_loss": train_loss,"acc": accuracy,"loss": loss}) 

            if verbose:
                print(f"Epoch {epoch+1}: train loss {train_loss}, val loss: {loss}, train acc {train_acc}, val acc: {accuracy}")
            else:
                pbar.update(1)# type: ignore
                pbar2.update(patience-pbar2.n) # type: ignore
                pbar.set_description(f"Epoch: {epoch+1}") # type: ignore
                pbar2.set_description(f"train_loss: {train_loss:.4f}, loss: {loss:.4f}, train_acc {train_acc:.4f}, acc: {accuracy:.4f}, Patience: ") # type: ignore
    if not verbose:
        pbar.close()# type: ignore
        pbar2.close()# type: ignore
    return net, optimizer, loss, accuracy # type: ignore

@blockPrintingIfServer
def train_shadow_model(target_model, 
                       shadow_model, 
                       trainloader, 
                       valloader, 
                       epochs: int, 
                       optimizer = None, 
                       criterion = None,  
                       device = get_device(), 
                       verbose=False, 
                       wandb_logging=False, 
                       accuracy_defined=False, 
                       patience= 5):
     
    #  Train a shadow model using a target model and a given dataset.
     
    #  Args:
    #      target_model (nn.Module): The target model to train the shadow model against.
    #      shadow_model (nn.Module): The shadow model to train.
    #      trainloader (DataLoader): The training data loader.
    #      valloader (DataLoader): The validation data loader.
    #      epochs (int): The number of epochs to train the shadow model.
    #      optimizer (torch.optim.Optimizer, optional): The optimizer to use for training. If None, Adam optimizer will be used.
    #      criterion (nn.Module, optional): The loss function to use. If None, MSELoss will be used.
    #      verbose (bool, optional): Whether to print verbose output during training. Default is False.
    #      wandb_logging (bool, optional): Whether to log training metrics to Weights & Biases. Default is True.
     
     
     

    if not criterion:
        criterion = torch.nn.MSELoss()
        # criterion = torch.nn.CrossEntropyLoss()
    if not optimizer:
        optimizer = torch.optim.Adam(shadow_model.parameters())
        # optimizer = optim.SGD(shadow_model.parameters(), lr=0.01)

    # # Split the dataset into training set and validation set
    # train_data, val_data = split_dataset(dataset)
    
    
    initial_patience = patience
    loss_min = 100000 # Inf
    # Training loop
    if not verbose:
        pbar = tqdm(total=epochs, position=1, leave=False)
        pbar2 = tqdm(total=patience, position=2, leave=False)
        pbar2.update(patience)
        pbar.set_description(f"Epoch {1}")
        pbar2.set_description(f"patience")
    for epoch in range(epochs):
        if patience<= 0:
                load_model(shadow_model, optimizer)
                val_loss = test_shadow_model(target_model, shadow_model, valloader, criterion, device)
                break
        else:
            # Training phase        
            shadow_model.train()
            target_model.eval()
            correct, total, train_loss = 0, 0, 0.0 
            # Forward pass through both the models
            # pdb.set_trace()
            for images, labels in trainloader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                expected_outputs = target_model(images)
                achieved_outputs = shadow_model(images)
                loss = criterion(achieved_outputs, expected_outputs)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()

                if accuracy_defined:  # Metrics
                    total += labels.size(0)
                    correct += (torch.max(achieved_outputs.data, 1)[1] == labels).sum().item()

            train_loss /= len(trainloader.dataset)
            if accuracy_defined:
                train_acc = correct / total

            # Validation phase
            val_loss, val_acc = test_shadow_model(target_model, shadow_model, valloader, criterion, device, accuracy_defined=accuracy_defined)

            if loss_min > val_loss:
                if verbose:
                    print(f"Patience reset")
                else:
                    pbar2.colour ="green" # type: ignore
                patience = initial_patience
                loss_min = val_loss
                save_model(shadow_model, optimizer)
            else:
                patience -= 1
                if not verbose:
                    pbar2.colour ="red" # type: ignore

            

        
        
        if wandb_logging:
            if accuracy_defined:
                wandb.log({"train_acc": train_acc, "train_loss": train_loss,"acc": val_acc,"loss": val_loss}) # type: ignore
            else:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        if verbose:
            print(f"Epoch {epoch+1}:train_loss: {train_loss:.4f}, Min loss: {loss_min:.4f}, Current loss: {val_loss:.4f}, train_acc {train_acc:.4f}, val_acc: {val_acc:.4f}, Patience: {patience}") # type: ignore
        else:
            pbar.update(1)# type: ignore
            pbar2.update(patience-pbar2.n) # type: ignore
            pbar.set_description(f"Epoch: {epoch+1}")# type: ignore
            if accuracy_defined:
                pbar2.set_description(f"train_loss: {train_loss:.4f}, loss: {val_loss:.4f}, train_acc {train_acc:.4f}, val_acc: {val_acc:.4f}, Patience: ") # type: ignore
            else:
                pbar2.set_description(f"train_loss: {train_loss:.4f}, loss: {val_loss:.4f}, Patience: ") # type: ignore
    if not verbose:
        pbar.close() # type: ignore

@blockPrintingIfServer
def test_shadow_model(target_model, shadow_model, testloader, criterion = None, device = get_device(), accuracy_defined=False):
    """Evaluate the model similirity on the entire test set."""
    if not criterion:
        # criterion = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.MSELoss()
 
    loss = 0.0
    target_model.eval()
    shadow_model.eval()
    acc = None
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            expected_outputs = target_model(images)
            achieved_outputs = shadow_model(images)
            loss_now = criterion(achieved_outputs, expected_outputs)
            loss += loss_now.item()   
            if accuracy_defined:  # Metrics
                total += labels.size(0)
                correct += (torch.max(achieved_outputs.data, 1)[1] == labels).sum().item()

        loss /= len(testloader.dataset)
        
        # pdb.set_trace()
        if accuracy_defined:
            acc = correct / total  
    return loss, acc

