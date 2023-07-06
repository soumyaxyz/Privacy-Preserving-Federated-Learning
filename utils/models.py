import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb, traceback

def load_model(model_name ="basic_CNN", num_channels=3, num_classes=10):
    # print (f'Loading model: {model_name}')
    if model_name =="basic_CNN":
        return basic_CNN(num_channels, num_classes)
    elif model_name == "basicCNN":
        return  basicCNN()
    elif model_name == "efficientnet":
        return  load_efficientnet(classes = num_classes)
    else:
        raise NotImplementedError

class basicCNN(nn.Module):
    def __init__(self) -> None:
        super(basicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 90
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class basic_CNN(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(basic_CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, 5) 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)        
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        try:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)      
            x = F.dropout(x, training=self.training)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)            
            return F.log_softmax(x, dim=1)
        except Exception as e:
                print(traceback.print_exc())
                pdb.set_trace()


def replace_classifying_layer(efficientnet_model, num_classes: int = 10):
    """Replaces the final layer of the classifier."""
    num_features = efficientnet_model.classifier.fc.in_features
    efficientnet_model.classifier.fc = torch.nn.Linear(num_features, num_classes)

def load_efficientnet(entrypoint: str = "nvidia_efficientnet_b0", classes: int = None):
    """Loads pretrained efficientnet model from torch hub. Replaces final
    classifying layer if classes is specified.

    Args:
        entrypoint: EfficientNet model to download.
                    For supported entrypoints, please refer
                    https://pytorch.org/hub/nvidia_deeplearningexamples_efficientnet/
        classes: Number of classes in final classifying layer. Leave as None to get the downloaded
                 model untouched.
    Returns:
        EfficientNet Model

    Note: One alternative implementation can be found at https://github.com/lukemelas/EfficientNet-PyTorch
    """
    efficientnet = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub", entrypoint, pretrained=True
    )

    if classes is not None:
        replace_classifying_layer(efficientnet, classes)
    return efficientnet