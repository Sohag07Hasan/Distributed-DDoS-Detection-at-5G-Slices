import flwr as fl
from collections import OrderedDict
from typing import Dict, List, Tuple
from flwr.common import NDArrays, Scalar
from utils import train, test, to_tensor, construct_autoencoder
from model import Net
import torch
from torch.utils.data import DataLoader
from config import (
    SERVER_ADDRESS, NUM_CLASSES, BATCH_SIZE, NUM_FEATURES, Q_PARAM
)
#from simulation import client_fn_callback
from flwr_datasets import FederatedDataset
#from dataloader import get_datasets, apply_transforms


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, client_id) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        self.model = construct_autoencoder(input_size=NUM_FEATURES)

        self.client_id = client_id #savign client ID

        self.q_param = Q_PARAM         

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
        lr, epochs = config["lr"], config["epochs"]
        
        #priting client info
        #print(f"[Client {self.client_id}] fit, config: {config}") 

        # Define the optimizer
        optim = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)

        global_weights = [param.clone().detach() for param in self.model.parameters()]

        # do local training
        train(self.model, self.trainloader, optim, epochs=epochs, device=self.device, q = self.q_param)

        # return the model parameters to the server as well as extra info (number of training examples in this case)
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model sent by the server on this client's
        local validation set. Then return performance metrics."""

        self.set_parameters(parameters)
        loss = test(self.model, self.valloader, device=self.device)
      
        return float(loss), len(self.valloader), {"loss": loss}


#Creates a lcient
def create_client(training_set, validation_set, client_id: int) -> fl.client.Client:

    # Now we apply the transform to each batch.
    trainloader = DataLoader(to_tensor(training_set), batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(to_tensor(validation_set), batch_size=BATCH_SIZE)
    
    # Create and return client
    return FlowerClient(trainloader, valloader, client_id).to_client()


if __name__ =="__main__":
    print('client.py')
