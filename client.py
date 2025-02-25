import flwr as fl
from collections import OrderedDict
from typing import Dict, List, Tuple
from flwr.common import NDArrays, Scalar
from utils import train, test, to_tensor, train_with_early_stopping, save_local_train_history_to_csv
from model import Net
import torch
from torch.utils.data import DataLoader
from config import SERVER_ADDRESS, NUM_CLASSES, BATCH_SIZE, NUM_FEATURES
#from simulation import client_fn_callback
from flwr_datasets import FederatedDataset
#from dataloader import get_datasets, apply_transforms


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, client_id) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net(input_size=NUM_FEATURES, num_classes=NUM_CLASSES)
        self.client_id = client_id #savign client ID

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def set_parameters(self, parameters):
        """With the model parameters received from the server,
        overwrite the uninitialise model in this class with them."""

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        # now replace the parameters
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract all model parameters and conver them to a list of
        NumPy arryas. The server doesn't work with PyTorch/TF/etc."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """This method train the model using the parameters sent by the
        server on the dataset of this client. At then end, the parameters
        of the locally trained model are communicated back to the server"""

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        # read from config
        lr, epochs, server_round = config["lr"], config["epochs"], config["server_round"]
        lr_adjustment_patience_on_epoch, lr_adjustment_factor, min_lr, early_stop_patience_on_epoch = config['lr_adjustment_patience_on_epoch'], config['lr_adjustment_factor'], config['min_lr'], config['early_stop_patience_on_epoch']
        alpha = config['alpha']


        # Define the optimizer
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        # do local training
        #train(self.model, self.trainloader, optim, epochs=epochs, device=self.device)
        #def train_with_early_stopping(client_id, net, trainloader, testloader, optimizer, epochs, device: str, patience=3, min_lr=0.001, max_lr=0.01):
        results = train_with_early_stopping(self.model, self.trainloader, self.valloader, optim, epochs=epochs, device=self.device, lr_patience=lr_adjustment_patience_on_epoch, early_stop_patience=early_stop_patience_on_epoch, factor=lr_adjustment_factor, min_lr=min_lr, alpha=alpha)
        parameters = list(results['model_state'].values())
        self.set_parameters(parameters) ## setting up the best model

        #Saving local model train history
        save_local_train_history_to_csv(self.client_id, server_round, results['metrics_history'])

        best_learing_rate = min(results['metrics_history']['learning_rate'])

        # planning to return the evaluation metrics with weights to make
        # utilizing best model performances
        loss, accuracy = test(self.model, self.valloader, device=self.device)

        # return the model parameters to the server as well as extra info (number of training examples in this case)
        return self.get_parameters({}), len(self.trainloader), {"accuracy": accuracy, "loss": float(loss), "num_eval_example": len(self.valloader), "client_id":self.client_id, "best_learning_rate": best_learing_rate} #"local_train_history": results['metrics_history']

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model sent by the server on this client's
        local validation set. Then return performance metrics."""

        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, device=self.device)

        # send statistics back to the server      
        return float(loss), len(self.valloader), {"accuracy": accuracy, "loss": float(loss), "client_id":self.client_id }


#Creates a lcient
def create_client(training_set, validation_set, client_id: int) -> fl.client.Client:

    # Now we apply the transform to each batch.
    trainloader = DataLoader(to_tensor(training_set), batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(to_tensor(validation_set), batch_size=32)
    
    # Create and return client
    return FlowerClient(trainloader, valloader, client_id).to_client()


if __name__ =="__main__":
    print('client.py')
