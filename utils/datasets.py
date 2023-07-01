import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

def load_CIFAR10():
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)

    return trainset, testset

def load_datasets(dataset_name = 'CIFAR10'):
    if dataset_name == 'CIFAR10':
        return trainset, testset = load_CIFAR10()
    else:
        raise NotImplementedError
    
