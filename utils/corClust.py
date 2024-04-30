import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, to_tree
import pdb, traceback

import torch
import torch.nn as nn

# A helper class for KitNET which performs a correlation-based incremental clustering of the dimensions in X
# n: the number of dimensions in the dataset
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection
class corClust:
    def __init__(self,n):
        #parameter:
        self.n = n
        #varaibles
        self.c = np.zeros(n) #linear num of features
        self.c_r = np.zeros(n) #linear sum of feature residules
        self.c_rs = np.zeros(n) #linear sum of feature residules
        self.C = np.zeros((n,n)) #partial correlation matrix
        self.N = 0 #number of updates performed

    # x: a numpy vector of length n
    def update(self,x):
        try:
            x = x.detach().cpu().numpy()
        except AttributeError:
            pass
        try:
            self.N += 1
            self.c += x
            c_rt = x - self.c/self.N
            self.c_r += c_rt
            self.c_rs += c_rt**2
            self.C += np.outer(c_rt,c_rt)
        except Exception as e:
            traceback.print_exc()
            pdb.set_trace()

    # creates the current correlation distance matrix between the features
    def corrDist(self):
        c_rs_sqrt = np.sqrt(self.c_rs)
        C_rs_sqrt = np.outer(c_rs_sqrt,c_rs_sqrt)
        C_rs_sqrt[C_rs_sqrt==0] = 1e-100 #this protects against dive by zero erros (occurs when a feature is a constant)
        D = 1-self.C/C_rs_sqrt #the correlation distance matrix
        D[D<0] = 0 #small negatives may appear due to the incremental fashion in which we update the mean. Therefore, we 'fix' them
        return D

    # clusters the features together, having no more than maxClust features per cluster
    def cluster(self,maxClust):
        D = self.corrDist()
        Z = linkage(D[np.triu_indices(self.n, 1)])  # create a linkage matrix based on the distance matrix
        if maxClust < 1:
            maxClust = 1
        if maxClust > self.n:
            maxClust = self.n
        map = self.__breakClust__(to_tree(Z),maxClust)
        return map

    # a recursive helper function which breaks down the dendrogram branches until all clusters have no more than maxClust elements
    def __breakClust__(self,dendro,maxClust):
        if dendro.count <= maxClust: #base case: we found a minimal cluster, so mark it
            return [dendro.pre_order()] #return the origional ids of the features in this cluster
        return self.__breakClust__(dendro.get_left(),maxClust) + self.__breakClust__(dendro.get_right(),maxClust)

class dA_params:
    def __init__(self, device = 'cpu',  n_visible = 5, n_hidden = 3, lr=0.001, corruption_level=0.0, gracePeriod = 10000, hiddenRatio=None, learning_rate=1e-4):
        self.device = device
        self.n_visible = n_visible# num of units in visible (input) layer
        self.n_hidden = n_hidden# num of units in hidden layer
        self.lr = lr
        self.corruption_level = corruption_level
        self.gracePeriod = gracePeriod
        self.hiddenRatio = hiddenRatio
        self.learning_rate = learning_rate
        if self.hiddenRatio is not None:
            self.n_hidden = int(np.ceil(self.n_visible*self.hiddenRatio))


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

class AE(nn.Module):
    def __init__(self, params):
        super(AE, self).__init__()
        # pdb.set_trace()
        self.encoder = nn.Linear(in_features=params.n_visible, out_features=params.n_hidden)
        self.decoder = nn.Linear(in_features=params.n_hidden, out_features=params.n_visible) 
    
    def forward(self, x):
        # x = F.relu(self.encoder(x))
        # pdb.set_trace()
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class dA:
    def __init__(self, params):     
        self.params = params
        # for 0-1 normlaization
        self.norm_max = torch.full((self.params.n_visible,), -float('inf'), device=params.device)
        self.norm_min = torch.full((self.params.n_visible,), float('inf'), device=params.device)


        # self.norm_max = np.ones((self.params.n_visible,)) * -np.Inf
        # self.norm_min = np.ones((self.params.n_visible,)) * np.Inf
        self.n = 0    # epoch / packet count

        try:
            self.model = AE(self.params).to(params.device)
        except Exception as e:
            traceback.print_exc()
            pdb.set_trace()
        self.trained = False


        torch.manual_seed(42)
        self.criterion = RMSELoss() # root mean square error loss
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate, weight_decay=1e-5) 
        # print('Adam')
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.learning_rate) 

        self.model.train()

    def get_corrupted_input(self, input, corruption_level):
        assert corruption_level < 1
        self.rng = np.random.RandomState(1234)
        return self.rng.binomial(size=input.shape, n=1, p=1 - corruption_level) * input

    def train(self, x): 
        x = torch.from_numpy(x).float().to(self.params.device)
        self.n = self.n + 1
        # update norms



        # self.norm_max[x > self.norm_max] = x[x > self.norm_max]
        # self.norm_min[x < self.norm_min] = x[x < self.norm_min]
        # Update norms: Here tensor operations will work directly since x, norm_max, and norm_min are all tensors on the same device
        self.norm_max = torch.max(self.norm_max, x)
        self.norm_min = torch.min(self.norm_min, x)

        # 0-1 normalize
        # x_normalized = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
        x_normalized = (x - self.norm_min) / (self.norm_max - self.norm_min + 1e-10)

        # x = x.astype(np.double)
        # x = torch.from_numpy(x)

        if self.params.corruption_level > 0.0:
            tilde_x = self.get_corrupted_input(x_normalized, self.params.corruption_level)
            # tilde_x = torch.from_numpy(tilde_x)
        else:
            tilde_x = x_normalized
            # tilde_x = torch.from_numpy(x)

        # x = torch.from_numpy(x)

        # pdb.set_trace()
        # self.model.double()
        x_r = self.model(tilde_x)
        loss = self.criterion(x_r, x)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.trained = True
        return loss.item() # RMSE

    def execute(self, x): #returns MSE of the reconstruction of x
        if self.n < self.params.gracePeriod:
            return 0.0
        else:
            assert self.trained, 'model not trained'
            self.model.eval()
            # 0-1 normalize
            x1 = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)

            self.model.double()
            x1 = torch.from_numpy(x1)

            x_r = self.model(x1)
            loss = self.criterion(x_r, x1)
            # pdb.set_trace()
            return loss.item() # RMSE















# Copyright (c) 2017 Yisroel Mirsky
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.