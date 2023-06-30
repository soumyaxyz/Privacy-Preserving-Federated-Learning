from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import flwr as fl
import wandb
import torch
from utils.client_utils import load_datasets
from utils.training_utils import wandb_init, get_device, get_parameters, train, test
from simulate import get_info
from utils.models import basicCNN as Net




def train_centralized(EPOCHS=50):
    trainloaders, valloaders, testloader, _ = load_datasets(num_clients = 1)


    trainloader = trainloaders[0]
    valloader = valloaders[0]
    net = Net().to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters())

    _, model_name, _ =  get_info(num_clients = 1, device=DEVICE, net= net)

    wandb_init("Centrailzed_"+model_name, model_name )


    net, optimizer = train(net, trainloader, valloader, EPOCHS, optimizer)
    loss, accuracy = test(net, testloader)
    wandb.log({"test_acc": accuracy,"test_loss": loss})
    print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")
    # wandb.finish()

if __name__ == "__main__":       

    DEVICE = get_device()  
    print(
        f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
    )


    EPOCHS = 5
    # create a main method
    for i in range(1):
        train_centralized(EPOCHS)