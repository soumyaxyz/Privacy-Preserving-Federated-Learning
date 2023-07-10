
import flwr
from tqdm import tqdm
from utils.training_utils import  get_parameters, save_model, load_model, delete_saved_model, set_parameters, test, train, train_single_epoch, wandb_init
import pdb,traceback, wandb
from pathlib import Path


def get_certificate():
        try:
            certificate=Path(".cache/certificates/ca.crt").read_bytes()
        except FileNotFoundError:
            print("Certificates not found. Falling back to unsecure mode")
            certificate = None
        return certificate

class FlowerClient(flwr.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, N , wandb_logging=False, dataset_name='CIFAR10', patience=5, simulation=False):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.wandb_logging = wandb_logging
        self.simulation = simulation
        self.patience = patience
        self.initial_patience = self.patience
        self.loss_min = 10000                   # inf 
        if patience > 9999: 
            self.ovefit_flag = "overfit"
        else:
            self.ovefit_flag = ''
        self.comment = 'Client_'+str(N)+'-'+str(cid)+'_'+self.ovefit_flag+'_'+net.__class__.__name__+'_'+dataset_name              
        if self.wandb_logging:            
            wandb_init(comment=self.comment, model_name=net.__class__.__name__, dataset_name=dataset_name)
        if not self.simulation:
            self.train_acc_bar = tqdm(total=1, position=1, leave=True)
            self.val_acc_bar = tqdm(total=1, position=2, leave=True)
            if not self.ovefit_flag:
                self.patience_bar = tqdm(total=self.patience, position=3, leave=True)

        
        # pdb.set_trace()

 

    def __del__(self):
        if not self.simulation:
            self.train_acc_bar.close()
            self.val_acc_bar.close()
            if not self.ovefit_flag:
                self.patience_bar.close()
                
        try:
            delete_saved_model(filename = self.comment)
        except:
            pdb.set_trace()
        if self.wandb_logging:
            wandb.finish()


    def get_parameters(self, config):
        # print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        # print(f"[Client {self.cid}] fit, config: {config}")
        set_parameters(self.net, parameters)
        if self.patience > 0:                                        # if patience is 0, stop further training
            if config["local_epochs"] == 1:
                self.loss, self.accuracy = train_single_epoch(self.net, self.trainloader)
            else:
                _, _, self.loss, self.accuracy, early_stopped = train(self.net, self.trainloader, self.valloader, epochs=config.local_epochs,  wandb_logging=False, patience= 2)
                if early_stopped:
                    self.patience = 0

        else:
            load_model(self.net, filename = self.comment)
        if self.wandb_logging:
            wandb.log({"train_acc": self.accuracy,"train_loss": self.loss})
            # print(f"[Client {self.cid}] fit(loss: {self.loss:.4f}, accuracy: {self.accuracy:.4f})")
            
        if not self.simulation:
            rounded_accuracy = round(self.accuracy, 4)
            self.train_acc_bar.update(rounded_accuracy-self.train_acc_bar.n) 
            self.train_acc_bar.set_description(f"Train_acc")
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        # print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        if self.patience > 0:                                    # if patience is 0, no point in evaluating
            if config["local_epochs"] == 1:                        # if training epoch > 1, use  validation results from last training epoch
                self.loss, self.accuracy = test(self.net, self.valloader)
            if not self.ovefit_flag:
                if self.loss_min > self.loss:                           # validation loss improved
                    self.patience = self.initial_patience               # reset patience
                    self.loss_min = self.loss 
                    save_model(self.net, filename = self.comment)
                else:
                    self.patience = max(self.patience-1, 0)             # decrease patience

        if not self.simulation:
            rounded_accuracy = round(self.accuracy, 4)
            self.val_acc_bar.update(rounded_accuracy-self.val_acc_bar.n)
            self.val_acc_bar.set_description("  Val_acc")
            if not self.ovefit_flag:
                self.patience_bar.update(self.patience-self.patience_bar.n)
                if self.patience > 0:
                    self.patience_bar.set_description("Patience")
                else:
                    self.patience_bar.set_description(f"Training stopped at client {self.cid} after server_round {config['server_round']} to prevent overfitting. Patience")
                    self.patience_bar.close()
        # print(f"[Client {self.cid}] evaluate(loss: {loss:.4f}, accuracy: {accuracy:.4f})")
        return float(self.loss), len(self.valloader), {"accuracy": float(self.accuracy)}


def client_fn(cid, net, trainloaders, valloaders, N=2, wandb_logging=False, dataset_name='CIFAR10', patience=5, simulation=False) -> FlowerClient:    
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader, N, wandb_logging, dataset_name, patience, simulation)