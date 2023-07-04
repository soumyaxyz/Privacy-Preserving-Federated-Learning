from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import numpy as np
import flwr as fl
import torch
from tqdm import tqdm
import wandb
import pdb,traceback

def wandb_init(
    project="Privacy-Preverving-Ferderated-Learning", 
    entity="soumyabanerjee", 
    model_name="CNN_1", 
    dataset_name="CIFAR_10",
    comment= '',  
    lr ='', 
    optimizer = ''
    ):    
    wandb.login()
    wandb.init(
      project=project, entity=entity,
      config={"learning_rate": lr, "optimiser": optimizer, "comment" : comment, "model": model_name, "dataset": dataset_name}
    )


def get_device():
    if torch.cuda.is_available():
      return torch.device("cuda")
    else:
      return torch.device("cpu")

def print_info(device, model_name="model", dataset_name="dataset"):
    print(f"Training on {model_name} with {dataset_name} in {device} using PyTorch {torch.__version__} and Flower {fl.__version__}")

def save_model(net, optim = None, filename ='filename'):
    path = './saved_models/'+filename+'.pt'
    if optim:
        torch.save({'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optim.state_dict()
            }, path)
    else:
        torch.save({'model_state_dict': net.state_dict()}, path)

def load_model(net, optim, filename ='filename'):
    path = './saved_models/'+filename+'.pt'
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])
    if optim:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

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

def train(net, trainloader, valloader, epochs: int, optimizer = None, criterion = None, verbose=False, wandb_logging=True):
    """Train the network on the training set."""
    if not criterion:
        criterion = torch.nn.CrossEntropyLoss()
    if not optimizer:
        optimizer = torch.optim.Adam(net.parameters())

    patience = 5
    loss_min = 100000 # Inf
    savefilename = net.__class__.__name__

    record_mode = False

    if not verbose:
        pbar = tqdm(total=epochs)
    for epoch in range(epochs):
        if record_mode:
            wandb.log({"acc": accuracy,"loss": loss})
            pbar.update(1)
        elif patience<= 0:
            try:
                
                load_model(net, optimizer, savefilename)
            except Exception as e:
                print(traceback.print_exc())
                pdb.set_trace()

            loss, accuracy = test(net, valloader)
            wandb.log({"acc": accuracy,"loss": loss}) 
            pbar.update(1)  
            pbar.set_description(f"Early stopped, t_loss: {train_loss:.4f}, loss: {loss:.4f}, t_acc {train_acc:.4f}, acc: {accuracy:.4f}")
            # break
            record_mode = True
        else:
            train_loss, train_acc = train_single_epoch(net, trainloader, optimizer, criterion) 
            loss, accuracy = test(net, valloader)
            if loss_min > loss: # validation loss improved
                patience = 5
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
                pbar.set_description(f"p: {patience}, t_loss: {train_loss:.4f}, loss: {loss:.4f}, t_acc {train_acc:.4f}, acc: {accuracy:.4f}")
    if not verbose:
        pbar.close()
    return net, optimizer

def train_shadow_model(target_model, shadow_model, trainloader, valloader, epochs: int, optimizer = None, criterion = None, verbose=False, wandb_logging=True):
     
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
    if not optimizer:
        optimizer = torch.optim.Adam(net.parameters())
        # optimizer = optim.SGD(shadow_model.parameters(), lr=0.01)

    # Split the dataset into training set and validation set
    train_data, val_data = split_dataset(dataset)
    
    patience = 5
    loss_min = 100000 # Inf
    # Training loop
    if not verbose:
        pbar = tqdm(total=epochs)
    for epoch in range(epochs):
        if patience<= 0:
                load_model(net, optimizer)
                val_loss = test_shadow_model(target_model, shadow_model, valloader, criterion, DEVICE)
                break
        else:
            # Training phase        
            shadow_model.train()
            target_model.eval()
            epoch_loss =  0.0
            # Forward pass through both the models
            for images, _ in trainloader:
                images = images.to(DEVICE)
                optimizer.zero_grad()
                expected_outputs = target_model(images)
                achieved_outputs = shadow_model(images)
                loss = criterion(achieved_outputs, expected_outputs)
                loss.backward()
                optimizer.step()
                # Metrics
                epoch_loss += loss
            epoch_loss /= len(trainloader.dataset)
            if loss_min > loss:
                patience = 0
                loss_min = loss
            else:
                patience -= 1

        # Validation phase
        val_loss = test_shadow_model(target_model, shadow_model, valloader, criterion, DEVICE)

        # Print the validation loss and metric
        print(f"Validation Loss: {val_loss.item()}, Metric: {val_metric}")
        
        if wandb_logging:
            wandb.log({"train_acc": train_acc, "train_loss": train_loss,"acc": accuracy,"loss": loss}) 

        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, val loss: {val_loss}")
        else:
            pbar.update(1)  
            pbar.set_description(f"p: {patience}, t_loss: {train_loss:.4f}, v_loss: {loss:.4f}")
    if not verbose:
        pbar.close()

def test_shadow_model(target_model, shadow_model, testloader, criterion = None, DEVICE = get_device()):
    """Evaluate the model similirity on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    loss = 0.0
    target_model.eval()
    shadow_model.eval()
    with torch.no_grad():
        for images, _ in testloader:
            images = images.to(DEVICE)
            expected_outputs = target_model(images)
            achieved_outputs = shadow_model(images)
            loss += criterion(achieved_outputs, expected_outputs).item()            
    loss /= len(testloader.dataset)
    return loss

