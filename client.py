import flwr as fl
from collections import OrderedDict
from typing import Dict, List, Tuple
from flwr.common import NDArrays, Scalar
from utils import train, test
from model import Net
import torch
from torch.utils.data import DataLoader
from config import SERVER_ADDRESS, NUM_CLASSES
#from simulation import client_fn_callback
from flwr_datasets import FederatedDataset
from dataloader import get_datasets, apply_transforms


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net(num_classes=NUM_CLASSES)
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

        # Define the optimizer
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        # do local training
        train(self.model, self.trainloader, optim, epochs=epochs, device=self.device)

        # return the model parameters to the server as well as extra info (number of training examples in this case)
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model sent by the server on this client's
        local validation set. Then return performance metrics."""

        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, device=self.device)
        # send statistics back to the server
        return float(loss), len(self.valloader), {"accuracy": accuracy}


#Creates a lcient
def create_client(dataset: FederatedDataset, cid: int) -> fl.client.Client:
    # Let's get the partition corresponding to the i-th client
    client_dataset = dataset.load_partition(int(cid), "train")

    # Now let's split it into train (90%) and validation (10%)
    client_dataset_splits = client_dataset.train_test_split(test_size=0.1, seed=42)

    trainset = client_dataset_splits["train"]
    valset = client_dataset_splits["test"]

    # Now we apply the transform to each batch.
    trainloader = DataLoader(
        trainset.with_transform(apply_transforms), batch_size=32, shuffle=True
    )
    valloader = DataLoader(valset.with_transform(apply_transforms), batch_size=32)

    # Create and return client
    return FlowerClient(trainloader, valloader).to_client()


if __name__ =="__main__":
    print('client.py')
