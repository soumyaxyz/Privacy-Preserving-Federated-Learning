from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import flwr as fl
import wandb
import torch
from utils.client_utils import load_datasets
from utils.training_utils import get_parameters, train, test
from utils.models import basicCNN as Net


DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)



def wandb_init(comment= '', lr ='', optimizer = '', model_name="CNN_1", dataset_name="CIFAR_10"):
    wandb.login()
    wandb.init(
      project="Ferderated-CIFAR_10", entity="soumyabanerjee",
      config={
        "learning_rate": lr,
        "optimiser": optimizer,
        "comment" : comment,
        "model": model_name,
        "dataset": dataset_name,
      }
    )


class Experiments(object):
    """docstring for Experiments"""
    def __init__(self, NUM_CLIENTS):
        super(Experiments, self).__init__()
        self.trainloaders, self.valloaders, self.testloader = load_datasets(NUM_CLIENTS)        

    # def client_fn(self, cid) -> FlowerClient:
    #     net = Net().to(DEVICE)
    #     trainloader = self.trainloaders[int(cid)]
    #     valloader = self.valloaders[int(cid)]
    #     return FlowerClient(cid, net, trainloader, valloader)


def train_centralized(EPOCHS=50):
    trainloaders, valloaders, testloader, _ = load_datasets(num_clients = 1)


    trainloader = trainloaders[0]
    valloader = valloaders[0]
    net = Net().to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters())

    wandb_init(comment= 'Centralized_CNN')


    net, optimizer = train(net, trainloader, valloader, EPOCHS, optimizer)


    loss, accuracy = test(net, testloader)
    wandb.log({"test_acc": accuracy,"test_loss": loss})
    print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")
    # wandb.finish()

# train_federated(NUM_CLIENTS=10)


NUM_CLIENTS=10
EPOCHS = 50



# create a main method
for i in range(1):
    train_centralized(EPOCHS)