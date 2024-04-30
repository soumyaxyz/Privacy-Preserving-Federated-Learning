from abc import ABC, abstractmethod
from typing import Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
import pdb, traceback
from sklearn.metrics import accuracy_score, mean_squared_error,  f1_score, precision_recall_curve, auc
from opacus.validators import ModuleValidator
import numpy as np
import scipy.cluster.hierarchy as sch
import inspect
import re
import wandb
from tqdm import tqdm

from utils.lib import blockPrinting
from utils.training_utils import load_pickle, sanitized_path, save_pickle
from utils import corClust as CC



def extract_model_names(func):
    source = inspect.getsource(func)
    model_names = set(re.findall(r'model_name == "(\w+)"', source))
    return model_names

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
    elif model_name == "kitnet_pt":
        cluster = Corr_Cluster_batch(num_channels)
        model = Kitnet( input_size=num_channels, cluster_module=cluster) 
    elif model_name == "AE":
        model = AE(input_size=num_channels)
    elif model_name == "lgb":
        # assert num_classes == 2
        # return Load_LGB() 
        raise NotImplementedError(f" Load {model_name} as LGB = Load_LGB(device) instead.")
    elif model_name == "CNN_malware":
        return CNN_malware(num_channels, num_classes)

    
    elif model_name == "attack_classifier":
        return binary_classifier(num_channels)
    else:
        raise NotImplementedError(f" {model_name} not defined yet")

    if differential_privacy:
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)            # Convert BatchNorm layers to GroupNorm
    
    return model


def load_non_pytorch_model_defination(model_name ="LGB",  device="cpu", wandb=False, num_channels=3, num_classes=10, differential_privacy=False):
    if model_name == "kitnet":
        # cluster = CC.corClust(num_channels)
        model = KitNET_OG( n=num_channels,device=device, wandb=wandb,   max_autoencoder_size=10,FM_grace_period=None,AD_grace_period=10000,learning_rate=0.1,hidden_ratio=0.75, feature_map = None)
    # elif model_name == "AE":
    #     model = AE(input_size=num_channels)
    elif model_name == "lgb":
        model = Load_LGB(device=device, wandb=wandb)
    else:
        raise NotImplementedError(f" {model_name} not defined yet")

    if differential_privacy:
        raise NotImplementedError(f" differential_privacy for {model_name} not defined yet")

    return model





class CNN_malware(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN_malware, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.dropout1 = nn.Dropout(0.4)
        self.bn1 = nn.BatchNorm1d(100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 100)
        self.dropout2 = nn.Dropout(0.4)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, num_classes)  # Output logits for each class

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x  # Output logits

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


class AbstractModelLoader(ABC):
    def __init__(self, device=None, wandb=False):
        self.device = device or 'cpu'  # default to CPU if no device is provided
        self.wandb_flag = wandb
        self.trained_model = None

    def save_model(self, filename, model=None):
        if model is None:
            assert self.trained_model is not None
            model = self.trained_model       

        filename = sanitized_path(filename)
        save_pickle(model, filename)
        print(f'Model saved to {filename}')

    def load_model(self, filename):
        # filename += self.param_id        # there is no need for this as the flag is already added in the filename
        filename = sanitized_path(filename)
        self.trained_model = load_pickle(filename)
        print(f'Model loaded from {filename}')

        return self.trained_model

    def wandb_callback(self):
        """Provide a callback for Weights & Biases, to be implemented by subclasses."""
        raise NotImplementedError("Please implement wandb_callback()")

    @abstractmethod
    def train(self, train_set, val_set, num_epochs: int) -> Any:
        """Train the model, to be implemented by subclasses."""
        pass

    @abstractmethod
    def predict(self, X_test, Y_test)-> Tuple[float, float, Any]:
        """Predict using the model, to be implemented by subclasses."""
        pass

    

    @abstractmethod
    def convert_data(self, X, y)-> Any:
        """Convert data to the necessary format, to be implemented by subclasses."""
        pass


class Load_LGB(AbstractModelLoader):
    def __init__(self, device=None, param_id = 'B', wandb=False):
        super(Load_LGB, self).__init__(device, wandb)        
        self.device = 'gpu' if str(device) == 'cuda' else str(device)
        self.params = self.get_param_by_id(param_id, self.device)
        self.param_id = 'B'

    def get_param_by_id(self, param_id, device='cpu'):
        print(f'Loading LGB with params {param_id}')
        if param_id == 'A':
            params = {
                'device_type':device,
                'num_leaves' : 10,
                'max_depth': 6,
                'learning_rate': 0.05,
                'objective': 'binary',
                'lambda_l2': 0.1, # Alias 'reg_lambda' L2 regularization
                'random_state': 42, 
                'verbosity': -1, 
                'metric': 'auc'
            }
            
        else:
            if param_id != 'B':
                print('Invalid param_id, defaulting to A')
            params = {
                'device_type':device,
                'num_leaves': 60,
                'min_data_in_leaf': 100, 
                'objective':'binary',
                'max_depth': -1,
                'learning_rate': 0.1,
                "boosting": "gbdt",
                "feature_fraction": 0.8,
                "bagging_freq": 1,
                "bagging_fraction": 0.8 ,
                "bagging_seed": 1,
                "metric": 'auc',
                "lambda_l1": 0.1,
                "random_state": 133,
                "verbosity": -1
            }              
        return params  
       
    def wandb_callback(self):
        import wandb
        def callback(env):
            """LightGBM to W&B callback"""
            # For each validation set, log metrics
            for i, eval_result in enumerate(env.evaluation_result_list):
                metric = eval_result[1]
                val_name = eval_result[0]
                score = eval_result[2]
                wandb.log({f"{val_name}/{metric}": score}, commit=False)
            wandb.log({"epoch": env.iteration})
        return callback

    def convert_data(self, X, y):
        # # Convert the datasets to LightGBM format
        lgb_dataset = lgb.Dataset(X, label=y)
        return lgb_dataset

    def train(self, lgb_train, lgb_val, num_epoch=100):
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)]
        if self.wandb_flag:
            callbacks.append(self.wandb_callback())

        if self.trained_model is None:
            self.trained_model = lgb.train(self.params, lgb_train, num_boost_round=num_epoch, valid_sets=[lgb_train, lgb_val], callbacks=callbacks)
        else:
            self.trained_model = lgb.train(self.params, lgb_train, num_boost_round=num_epoch, valid_sets=[lgb_train, lgb_val], callbacks=callbacks, init_model=self.trained_model)
        return self.trained_model
    
    def predict(self,  X_test, Y_test):
        assert self.trained_model is not None
        test_pred = self.trained_model.predict(X_test)        
        threshold = 0.5
        test_pred_labels = (test_pred > threshold).astype(int) # type: ignore
        accuracy = accuracy_score(Y_test, test_pred_labels)
        loss = mean_squared_error(Y_test, test_pred) # type: ignore
        return loss, accuracy, test_pred

    def save_model(self, savefilename):        
        savefilename += self.param_id
        super().save_model(savefilename)

    
    

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_ratio, hidden_size=None):
        super(AutoEncoder, self).__init__()
        if hidden_size is None:
            hidden_size = max(3, int(input_size * hidden_ratio))
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        encoded = F.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded


class AE(AutoEncoder):
    def __init__(self, input_size, hidden_ratio=.5, hidden_size=None):
        super(AE, self).__init__(input_size, hidden_ratio, hidden_size)
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, x):
        decoded = super(AE, self).forward(x)
        loss_per_element = self.criterion(decoded, x)
        loss_per_sample = torch.mean(loss_per_element, dim=1, keepdim=True)
        log_loss = torch.log(loss_per_sample + 1)
        tanh_mapped = torch.tanh(log_loss)
        final_mapped_loss = (tanh_mapped + 1) / 2

        # pdb.set_trace()

        return final_mapped_loss

class Corr_Cluster_batch():
    def __init__(self, n):
        self.n = n
        self.c = np.zeros(n)  # Cumulative sum of features
        self.c_r = np.zeros(n)  # Residuals sum
        self.c_rs = np.zeros(n)  # Sum of squared residuals
        self.C = np.zeros((n, n))  # Correlation matrix (actually covariance-like)
        self.N = 0  # Total number of data points processed

    def update(self, batch):
        # Check if input batch is a PyTorch tensor and convert it to NumPy array if necessary
        if isinstance(batch, torch.Tensor):
            batch = batch.detach().cpu().numpy()
        # Assume batch is a NumPy array of shape [batch_size, n]
        batch_size = batch.shape[0]
        self.N += batch_size
        batch_sum = np.sum(batch, axis=0)
        self.c += batch_sum
        
        # Update residuals
        mean = self.c / self.N
        c_rt = batch - mean
        self.c_r += np.sum(c_rt, axis=0)
        self.c_rs += np.sum(c_rt**2, axis=0)
        
        # Update correlation matrix
        for i in range(batch_size):
            self.C += np.outer(c_rt[i], c_rt[i])

    def corrDist(self):
        # Compute the correlation distance matrix
        c_rs_sqrt = np.sqrt(self.c_rs)
        C_rs_sqrt = np.outer(c_rs_sqrt, c_rs_sqrt)
        C_rs_sqrt[C_rs_sqrt == 0] = 1e-10  # Prevent division by zero
        D = 1 - self.C / C_rs_sqrt
        D[D < 0] = 0  # Correct for numerical instability
        return D

    def cluster(self, maxClust):
        # Cluster features based on the correlation distance matrix
        D = self.corrDist()
        triu_idx = np.triu_indices(self.n, 1)
        Z = sch.linkage(D[triu_idx], method='average')
        clusters = sch.fcluster(Z, maxClust, criterion='maxclust')
        return {i: np.where(clusters == i + 1)[0] for i in range(maxClust)}



class Kitnet(nn.Module):
    def __init__(self, input_size, cluster_module, hidden_ratio=.5, clustering_batches=10, max_autoencoder_size=10):
        super(Kitnet, self).__init__()
        self.input_size = input_size
        self.hidden_ratio = hidden_ratio
        self.cluster_module = cluster_module
        self.autoencoders = nn.ModuleList()
        self.error_autoencoder = AutoEncoder(input_size, self.hidden_ratio)  #placeholder  to be replaced by initialized_autoencoders()
        self.m = max_autoencoder_size
        self.clustering_batches = clustering_batches
        self.batch_count = 0
        self.initialized = False


    def initialize_autoencoders(self, feature_clusters, device):
        try:
            for feature_indices in feature_clusters.values():
                if feature_indices.any():
                    feature_size = len(feature_indices)
                    self.autoencoders.append(AutoEncoder(feature_size, self.hidden_ratio).to(device)) 
            self.error_autoencoder = AutoEncoder(len(self.autoencoders), self.hidden_ratio).to(device)
            self.initialized = True
        except:
            traceback.print_exc()
            pdb.set_trace()

    def forward(self, x):
        try:
            if not self.initialized:
                self.cluster_module.update(x)  # Update clustering with batch
                if self.batch_count >= self.clustering_batches:
                    feature_clusters = self.cluster_module.cluster(self.m)
                    self.initialize_autoencoders(feature_clusters, x.device)
                    self.initialized = True
                self.batch_count += 1
                return torch.zeros(len(x), dtype=x.dtype, device=x.device, requires_grad=True)  # Temporary return during initialization
            
            
            reconstructed = torch.zeros_like(x).to(x.device)
            feature_clusters = self.cluster_module.cluster(self.m)  # Obtain current feature clusters
            
            reconstruction_errors = []
            for idx, autoencoder in enumerate(self.autoencoders):
                feature_indices = list(feature_clusters[idx])
                # mask = torch.zeros_like(x, dtype=torch.bool)
                # mask[:, feature_indices] = True

                if feature_indices:  # Check if there are features to process
                    reconstructed_part = autoencoder(x[:, feature_indices])
                    error = torch.sqrt(((x[:, feature_indices] - reconstructed_part) ** 2).mean(dim=1))
                    reconstruction_errors.append(error)
                    # print(error.shape)
                    # reconstructed[:, feature_indices] = reconstructed_part
                
                # reconstructed_part = autoencoder(x[:, feature_indices])
                # padding = (0, reconstructed.shape[1] - reconstructed_part.shape[1])  # pad the second dimension
                # reconstructed_part_padded = F.pad(reconstructed_part, pad=padding, mode='constant', value=0)
                # reconstructed = torch.where(mask, reconstructed_part_padded, reconstructed)
                
            reconstruction_error = torch.stack(reconstruction_errors).T
            

            error_reconstruction = self.error_autoencoder(reconstruction_error) 

            # Calculate final error by some reduction
            # final_error = (reconstruction_error.T.sum(0) - error_reconstruction.squeeze()).pow(2).mean(dim=0)

            final_error = torch.sqrt((reconstruction_error - error_reconstruction).pow(2).mean(1))



            # # Compute RMSE per feature
            # reconstruction_error = (x - reconstructed).pow(2).mean(dim=0).sqrt()
            # # Error autoencoder aggregates RMSEs into a single error per sample
            # error_reconstruction = self.error_autoencoder(reconstruction_error.unsqueeze(0))        
            # # Calculate final error
            # final_error = (reconstruction_error - error_reconstruction.squeeze()).pow(2).mean()

            # # reconstruction_error = (x - reconstructed).pow(2).sqrt()        
            # # error_reconstruction = self.error_autoencoder(reconstruction_error.unsqueeze(1))
            # # final_error = (reconstruction_error.unsqueeze(1) - error_reconstruction).pow(2).sum()

            # Apply sigmoid to the final error to map it to [0, 1]
            final_prob = torch.sigmoid(final_error)
            # pdb.set_trace()
        except:
            traceback.print_exc()
            pdb.set_trace()

        return final_prob



class KitNET_OG(AbstractModelLoader):
    #n: the number of features in your input dataset (i.e., x \in R^n)
    #m: the maximum size of any autoencoder in the ensemble layer
    #AD_grace_period: the number of instances the network will learn from before producing anomaly scores
    #FM_grace_period: the number of instances which will be taken to learn the feature mapping. If 'None', then FM_grace_period=AM_grace_period
    #learning_rate: the default stochastic gradient descent learning rate for all autoencoders in the KitNET instance.
    #hidden_ratio: the default ratio of hidden to visible neurons. E.g., 0.75 will cause roughly a 25% compression in the hidden layer.
    #feature_map: One may optionally provide a feature map instead of learning one. The map must be a list,
    #           where the i-th entry contains a list of the feature indices to be assingned to the i-th autoencoder in the ensemble.
    #           For example, [[2,5,3],[4,0,1],[6,7]]
    def __init__(self,n, device, wandb, max_autoencoder_size=10,FM_grace_period=None,AD_grace_period=10000,learning_rate=0.1,hidden_ratio=0.75, feature_map = None):
        super(KitNET_OG, self).__init__(device, wandb)
        self.device = "cpu" # override the device as batch processing is not supported
        # Parameters:
        self.AD_grace_period = AD_grace_period
        if FM_grace_period is None:
            self.FM_grace_period = AD_grace_period
        else:
            self.FM_grace_period = FM_grace_period
        if max_autoencoder_size <= 0:
            self.m = 1
        else:
            self.m = max_autoencoder_size
        self.lr = learning_rate
        self.hr = hidden_ratio
        self.n = n
        self.msg = ''

        # Variables
        self.n_trained = 0 # the number of training instances so far
        self.n_executed = 0 # the number of executed instances so far
        self.v = feature_map
        if self.v is None:
            # print("\nFeature-Mapper: train-mode, Anomaly-Detector: off-mode")
            pass
        else:
            self.__createAD__()
            # print("\nFeature-Mapper: execute-mode, Anomaly-Detector: train-mode")
        self.FM = CC.corClust(self.n) #incremental feature cluatering for the feature mapping process

        self.threshold = None
        self.ensembleLayer = []
        self.outputLayer = None



    #If FM_grace_period+AM_grace_period has passed, then this function executes KitNET on x. Otherwise, this function learns from x.
    #x: a numpy array of length n
    #Note: KitNET automatically performs 0-1 normalization on all attributes.
    def process_one(self,x):
        self.msg = ''
        if self.n_trained > self.FM_grace_period + self.AD_grace_period: #If both the FM and AD are in execute-mode
            return self.predict_one(x), self.msg 
        else:
            score = self.train_one(x)
            # print(self.m)
            # return 0.0, self.msg
            return score , self.msg

    #force train KitNET on x
    #returns the anomaly score of x during training (do not use for alerting)
    def train_one(self,x):
        score = 0
        if self.n_trained <= self.FM_grace_period and self.v is None: #If the FM is in train-mode, and the user has not supplied a feature mapping
            #update the incremetnal correlation matrix
            # pdb.set_trace()
            self.FM.update(x)
            # if self.n_trained <self.FM_grace_period:
                # if self.n_trained%10 ==0:
                #     print(self.FM.cluster(self.m))


            if self.n_trained == self.FM_grace_period: #If the feature mapping should be instantiated
                self.v = self.FM.cluster(self.m)
                # print(self.v)
                # import sys
                # sys.exit()
                self.__createAD__()
                # print("The Feature-Mapper found a mapping: "+str(self.n)+" features to "+str(len(self.v))+" autoencoders.")
                # print("Feature-Mapper: execute-mode, Anomaly-Detector: train-mode")
                self.msg = "The Feature-Mapper found a mapping: "+str(self.n)+" features to "+str(len(self.v))+" autoencoders."
                self.msg +="\nFeature-Mapper: execute-mode, Anomaly-Detector: train-mode"
        else: #train
            ## Ensemble Layer
            S_l1 = np.zeros(len(self.ensembleLayer))
            for a in range(len(self.ensembleLayer)):
                # make sub instance for autoencoder 'a'
                xi = x[self.v[a]] # type: ignore
                S_l1[a] = self.ensembleLayer[a].train(xi)
            ## OutputLayer
            score = self.outputLayer.train(S_l1) # type: ignore
            if self.n_trained == self.AD_grace_period+self.FM_grace_period:
                self.msg = "Feature-Mapper: execute-mode, Anomaly-Detector: execute-mode"
                # print("Feature-Mapper: execute-mode, Anomaly-Detector: execute-mode")
        self.n_trained += 1
        return score

    #force execute KitNET on x
    def predict_one(self,x):
        if self.v is None:
            raise RuntimeError('KitNET Cannot execute x, because a feature mapping has not yet been learned or provided. Try running process(x) instead.')
        else:
            # pdb.set_trace()
            self.n_executed += 1
            ## Ensemble Layer
            S_l1 = np.zeros(len(self.ensembleLayer))
            for a in range(len(self.ensembleLayer)):
                # make sub inst
                xi = x[self.v[a]]
                S_l1[a] = self.ensembleLayer[a].execute(xi)
            ## OutputLayer
            # pdb.set_trace()
            return self.outputLayer.execute(S_l1) # type: ignore

    def __createAD__(self):
        # pdb.set_trace()
        # construct ensemble layer
        for map in self.v: # type: ignore
            params = CC.dA_params(device=self.device, n_visible=len(map), n_hidden=0, lr=self.lr, corruption_level=0, gracePeriod=0, hiddenRatio=self.hr) 
            self.ensembleLayer.append(CC.dA(params)) # type: ignore

        # construct output layer
        params = CC.dA_params(device=self.device, n_visible=len(self.v), n_hidden=0, lr=self.lr, corruption_level=0, gracePeriod=0, hiddenRatio=self.hr) # type: ignore
        self.outputLayer = CC.dA(params) # type: ignore

    def define_threshold(self, scores, labels):
        """Define an optimal threshold based on the scores and true labels."""
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        # Calculate F1 scores for each possible threshold
        f1_scores = 2 * (precision * recall) / (precision + recall)
        # Locate the index of the largest F1 score
        optimal_idx = np.argmax(f1_scores)
        self.threshold = thresholds[optimal_idx]
        print(f"Optimal threshold set to: {self.threshold}")

    def convert_data(self, X, y):
        return (X, y)
        
    
    def train(self, train, val=None, num_epochs=1):   
        train_x, train_y = train     
        scores = np.zeros(len(train_x))  # Array to store scores for each sample in the batch
        
        
        for idx, x in tqdm(enumerate(train_x), total=len(train_x), desc="Training"):
            scores[idx] = self.train_one(x)
            if self.wandb_flag:            
                wandb.log({"anomaly_scores": scores})
        
        self.define_threshold(scores, train_y)

        return scores

    def predict(self, test_x, test_y):
        assert self.threshold is not None, "Threshold must be defined before prediction"

        scores = np.zeros(len(test_x))  # Array to store scores for each sample in the batch
        for idx, x in tqdm(enumerate(test_x), total=len(test_x), desc="Predicting"):
            scores[idx] = self.predict_one(x)

        # Convert scores to binary predictions based on the threshold
        test_pred = (scores > self.threshold).astype(int)
        
        # Calculate accuracy
        accuracy = accuracy_score(test_y, test_pred)
        
        # Calculate mean squared error as a form of 'loss'
        loss = mean_squared_error(test_y, scores)  # Note: this 'loss' calculation is not typical for classification tasks
        
        return loss, accuracy, scores



















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
