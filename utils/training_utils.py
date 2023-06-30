from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import numpy as np
import flwr as fl
import torch
from tqdm import tqdm
import wandb

def wandb_init(
    project="Ferderated-CIFAR_10", 
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

def print_info(device):
    print(f"Training on {device} using PyTorch {torch.__version__} and Flower {fl.__version__}")

def save_model(net, optim, path ='./cifar_net.pth'):
    torch.save({'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optim.state_dict()
            }, path)

def load_model(net, optim, path ='./cifar_net.pth'):
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])
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
        epoch_loss += loss
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

def train(net, trainloader, valloader, epochs: int, optimizer = None, criterion = None, verbose=True, wandb_logging=True):
    """Train the network on the training set."""
    if not criterion:
        criterion = torch.nn.CrossEntropyLoss()
    if not optimizer:
        optimizer = torch.optim.Adam(net.parameters())

    patience = 5
    loss_min = 100000 # Inf

    if not verbose:
        iterrator = tqdm(range(epochs))
    else:
        iterrator = range(epochs)
    for epoch in iterrator:
        if patience<= 0:
                load_model(net, optimizer)
                loss, accuracy = test(net, valloader)
                break
        else:
            train_loss, train_acc = train_single_epoch(net, trainloader, optimizer, criterion) 
            loss, accuracy = test(net, valloader)

        if wandb_logging:
            wandb.log({"train_acc": train_acc, "train_loss": train_loss,"acc": accuracy,"loss": loss})
            
        if verbose:
            print(f"Epoch {epoch+1}: train loss {train_loss}, val loss: {loss}, train acc {train_acc}, val acc: {accuracy}")
    return net, optimizer