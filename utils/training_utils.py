from collections import OrderedDict
from typing import Dict, List
import numpy as np
import flwr as fl
import torch
from tqdm import tqdm
import wandb
import pdb,traceback
import os

def wandb_init(
    project="Privacy-Preverving-Ferderated-Learning", 
    entity="soumyabanerjee", 
    model_name="basicCNN", 
    dataset_name="CIFAR_10",
    comment= '',  
    lr ='', 
    optimizer = ''
    ):    
    wandb.login( key="6a2ec88cea517d22c5c4db178898f7143e8a6ef3" )
    wandb.init(
      project=project, entity=entity,
      config={"learning_rate": lr, "optimiser": optimizer, "comment" : comment, "model": model_name, "dataset": dataset_name}
    )


def get_device():
    if torch.cuda.is_available():
      return torch.device("cuda")
    else:
      return torch.device("cpu")

def print_info(device, model_name="model", dataset_name="dataset", teacher_name=None):
    if device.type=="cuda":
        device_type = torch.cuda.get_device_name(0)  
    else:
        device_type = device.type
        
    if teacher_name:
        print(f"Distiling on {model_name} from {teacher_name} on {dataset_name} in {device_type} using PyTorch {torch.__version__}")
    else:
        print(f"Training on {model_name} with {dataset_name} in {device_type} using PyTorch {torch.__version__} and Flower {fl.__version__}")

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
        print(f"Saved model to {path}")

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

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def train_single_epoch(net, trainloader, optimizer = None, criterion = None, DEVICE = get_device()):
    """Train the network on the training set."""
    if not criterion:
        criterion = torch.nn.CrossEntropyLoss()
    if not optimizer:
        optimizer = torch.optim.Adam(net.parameters())
    net.train()
    correct, total, epoch_loss = 0, 0, 0.0
    for images, labels in trainloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(net(images), labels)
        loss.backward()
        optimizer.step()
        # Metrics
        epoch_loss += loss.item()
        total += labels.size(0)
        correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    epoch_loss /= len(trainloader.dataset)
    epoch_acc = correct / total
    # print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
    return epoch_loss, epoch_acc

def test(net, testloader, DEVICE = get_device() ):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def train(net, trainloader, valloader, epochs: int, optimizer = None, criterion = None, verbose=False, wandb_logging=True, patience= 5, loss_min = 100000):
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
                wandb.log({"acc": accuracy,"loss": loss})
            pbar.update(1)
        elif patience<= 0:
            try:
                load_model(net, optimizer, savefilename)
            except Exception as e:
                print(traceback.print_exc())
                pdb.set_trace()
            loss, accuracy = test(net, valloader)
            if wandb_logging:
                wandb.log({"acc": accuracy,"loss": loss}) 
            if not verbose:
                pbar.update(1)  
                pbar2.set_description(f"Early stopped at epoch {epoch+1}, train_loss: {train_loss:.4f}, loss: {loss:.4f}, train_acc {train_acc:.4f}, acc: {accuracy:.4f}")
            # break
            record_mode = True
        else:
            train_loss, train_acc =  train_single_epoch(net, trainloader, optimizer, criterion) 
            loss, accuracy = test(net, valloader)

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
                pbar.update(1)
                pbar2.update(patience-pbar2.n) 
                pbar.set_description(f"Epoch: {epoch+1}")
                pbar2.set_description(f"train_loss: {train_loss:.4f}, loss: {loss:.4f}, train_acc {train_acc:.4f}, acc: {accuracy:.4f}, Patience: ")
    if not verbose:
        pbar.close()
        pbar2.close()
    return net, optimizer, loss, accuracy 

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
                    pbar2.colour ="green"
                patience = initial_patience
                loss_min = val_loss
                save_model(shadow_model, optimizer)
            else:
                patience -= 1
                if not verbose:
                    pbar2.colour ="red"

            

        
        
        if wandb_logging:
            if accuracy_defined:
                wandb.log({"train_acc": train_acc, "train_loss": train_loss,"acc": val_acc,"loss": val_loss})
            else:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        if verbose:
            print(f"Epoch {epoch+1}:train_loss: {train_loss:.4f}, Min loss: {loss_min:.4f}, Current loss: {val_loss:.4f}, train_acc {train_acc:.4f}, val_acc: {val_acc:.4f}, Patience: {patience}")
        else:
            pbar.update(1)
            pbar2.update(patience-pbar2.n) 
            pbar.set_description(f"Epoch: {epoch+1}")
            if accuracy_defined:
                pbar2.set_description(f"train_loss: {train_loss:.4f}, loss: {val_loss:.4f}, train_acc {train_acc:.4f}, val_acc: {val_acc:.4f}, Patience: ")
            else:
                pbar2.set_description(f"train_loss: {train_loss:.4f}, loss: {val_loss:.4f}, Patience: ")
    if not verbose:
        pbar.close()

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

