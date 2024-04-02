import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
import pdb, traceback
from opacus.validators import ModuleValidator
from utils.lib import blockPrinting

@blockPrinting
def load_model_defination(model_name ="basic_CNN", num_channels=3, num_classes=10, differential_privacy=False):
    # print (f'Loading model: {model_name}')
    if model_name =="basic_CNN":
        model = basic_CNN(num_channels, num_classes)
    elif model_name == "basicCNN_CIFAR100":
        model = basicCNN_CIFAR100()
    elif model_name == "basicCNN_MNIST":
        model =  basicCNN_MNIST()
    elif model_name == "basicCNN":
        model =  basicCNN()
    elif model_name == "efficientnet":
        model =  load_efficientnet(classes = num_classes)
    elif model_name == "resnet":
        model = load_resnet(classes= num_classes)
    elif model_name == "googlenet":
        model = load_googlenet(classes= num_classes)
    elif model_name == "resnext":
        model = load_resnext(classes = num_classes) 
    elif model_name == "densenet":
        model = load_densenet(classes= num_classes)
    elif model_name == "mobilenet":
        model = load_mobilenet(classes= num_classes)
    elif model_name == "vgg":
        model = load_vgg(classes= num_classes)
    elif model_name == "inception":
        model = load_inception(classes= num_classes)
    elif model_name == "squeezenet":
        model = load_squeezenet(classes= num_classes)
    elif model_name == "shufflenet":
        model = load_shufflenet(classes= num_classes)
    elif model_name == "alexnet":
        model = load_alexnet (classes = num_classes)
    elif model_name == "lgb":
        assert num_classes == 2
        return Load_LGB() 
    elif model_name == "attack_classifier":
        return binary_classifier(num_channels)
    else:
        raise NotImplementedError(f" {model_name} not defined yet")

    if differential_privacy:
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)            # Convert BatchNorm layers to GroupNorm
    
    return model


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

class basicCNN_CIFAR100(nn.Module):
    def __init__(self) -> None:
        super(basicCNN_CIFAR100, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # Increase the number of output channels for conv1
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # Increase the number of output channels for conv2
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # Add a third convolutional layer
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)  # Adjust the size of fc1
        self.fc2 = nn.Linear(256, 128)  # Adjust the size of fc2
        self.fc3 = nn.Linear(128, 100)  # Changed the output size to 100 for CIFAR-100

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))  # Add a third convolutional layer
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class basicCNN_MNIST(nn.Module):
    def __init__(self) -> None:
        super(basicCNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 90
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
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

class binary_classifier(nn.Module):
    def __init__(self,input_shape):
        super(binary_classifier,self).__init__()
        self.fc1 = nn.Linear(input_shape,32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64,1)  

    def forward(self,x):
        try:
            x = torch.unsqueeze(x, 1).to(self.fc1.weight.dtype)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
            x = torch.squeeze(x)
        except Exception as e:
            traceback.print_exc()
            pdb.set_trace()
        return x
    
class Load_LGB:
    def __init__(self):
        self.params = {
        'device_type':'cpu',
        'num_leaves' : 10,
        'max_depth': 6,
        'learning_rate': 0.05,
        'objective': 'binary',
        'lambda_l2': 0.1, # Alias 'reg_lambda' L2 regularization
        'random_state': 42, 
        'verbosity': -1, 
        'metric': 'auc'
        }

    def convert_data(self, X, y):
        # # Convert the datasets to LightGBM format
        lgb_dataset = lgb.Dataset(X, label=y)
        return lgb_dataset

    def train(self, lgb_train, lgb_val, num_boost_round=100):
        trained_model = lgb.train(self.params, lgb_train, num_boost_round=num_boost_round, valid_sets=[lgb_train, lgb_val], callbacks=[lgb.early_stopping(200), lgb.log_evaluation(10)])
        
        return trained_model
    
    def predict(self, model, X_test):
        return model.predict(X_test)
    



def replace_classifying_layer(efficientnet_model, num_classes: int = 10):
    """Replaces the final layer of the classifier."""
    num_features = efficientnet_model.classifier.fc.in_features
    efficientnet_model.classifier.fc = torch.nn.Linear(num_features, num_classes)

def load_efficientnet(entrypoint: str = "nvidia_efficientnet_b0", classes: int = None): # type: ignore
    efficientnet = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", entrypoint, pretrained=True)
    if classes is not None:
        replace_classifying_layer(efficientnet, classes)
    return efficientnet

def load_resnet(entrypoint: str = "resnet18", classes:int = None):# type: ignore
    resnet = torch.hub.load('pytorch/vision:v0.10.0', entrypoint, pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    return resnet


def load_googlenet(entrypoint: str = "googlenet", classes:int = None):# type: ignore
    googlenet = torch.hub.load('pytorch/vision:v0.10.0', entrypoint, pretrained=True)
    return googlenet


def load_resnext(entrypoint: str = "resnext50_32x4d", classes:int = None): # type: ignore
    resnext = torch.hub.load('pytorch/vision:v0.10.0', entrypoint, pretrained=True)

    return resnext

def load_densenet(entrypoint: str = "densenet121", classes:int = None):# type: ignore
    densenet = torch.hub.load('pytorch/vision:v0.10.0', entrypoint, pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet169', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet161', pretrained=True)
    return densenet

def load_mobilenet(entrypoint: str = "mobilenet_v2", classes:int = None):# type: ignore
    mobilenet = torch.hub.load('pytorch/vision:v0.10.0', entrypoint, pretrained=True)
    return mobilenet

def load_vgg(entrypoint: str = "vgg11", classes:int = None):# type: ignore
    vgg = torch.hub.load('pytorch/vision:v0.10.0', entrypoint, pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg13', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg13_bn', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True)
    return vgg

def load_inception(entrypoint: str = "inception_v3", classes:int = None):# type: ignore
    inception = torch.hub.load('pytorch/vision:v0.10.0', entrypoint, pretrained=True)
    return inception

def load_squeezenet(entrypoint: str = "squeezenet1_0", classes:int = None):# type: ignore
    squeezenet = torch.hub.load('pytorch/vision:v0.10.0', entrypoint, pretrained=True)
    return squeezenet

def load_shufflenet (entrypoint: str = "shufflenet_v2_x1_0", classes:int = None):# type: ignore
    shufflenet = torch.hub.load('pytorch/vision:v0.10.0', entrypoint, pretrained=True)
    return shufflenet

def load_alexnet (entrypoint: str = "alexnet", classes:int = None):# type: ignore
    alexnet = torch.hub.load('pytorch/vision:v0.10.0', entrypoint, pretrained=True)
    return alexnet