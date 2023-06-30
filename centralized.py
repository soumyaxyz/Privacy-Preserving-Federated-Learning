from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import flwr as fl
import wandb
import torch
from utils.client_utils import load_datasets
from utils.training_utils import wandb_init,  print_info, get_device, get_parameters, train, test
from simulate import get_info
from utils.models import basicCNN as Net




def train_centralized(EPOCHS=50, DEVICE="cpu", wandb_logging=True):
    trainloaders, valloaders, testloader, _ = load_datasets(num_clients = 1)


    trainloader = trainloaders[0]
    valloader = valloaders[0]
    net = Net().to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters())

    _, model_name, _ =  get_info(num_clients = 1, device=DEVICE, net= net)
    if wandb_logging:
        wandb_init(comment="Centrailzed_"+model_name, model_name=model_name )

    
    net, optimizer = train(net, trainloader, valloader, EPOCHS, optimizer, wandb_logging=wandb_logging)
    loss, accuracy = test(net, testloader)

    if wandb_logging:
        wandb.log({"test_acc": accuracy,"test_loss": loss})
        wandb.finish()
    print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")
     
def main():
    DEVICE = get_device()  
    print_info(DEVICE)
    wandb_logging = True
    EPOCHS = 50
    repeat_expriment = 1
    for i in range(repeat_expriment):
        train_centralized(EPOCHS, DEVICE, wandb_logging)


if __name__ == "__main__":       
    main()
    