''' 
We'll be training the model in a Federated setting. In order to do that, we need to define two functions:

* `train()` that will train the model given a dataloader.
* `test()` that will be used to evaluate the performance of the model on held-out data, e.g., a training set.
'''
from config import (
    NUM_ROUNDS, GLOBAL_MODEL_PATH, NUM_CLASSES, BATCH_SIZE, FOLDER_NAME, 
    FOLD, NUM_FEATURES, FEATURE_TYPE, AUTOENCODER_LAYERS
)
from model import AutoEncoder
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd



## This function will train the model with early stopping implemented
def construct_autoencoder(input_size=NUM_FEATURES):
    df = pd.read_csv(AUTOENCODER_LAYERS)
    row = df[df["input_size"] == input_size].iloc[0]

    # Extract parameters
    hidden_sizes = row["hidden_sizes"]  # String, will be converted inside the class
    latent_size = int(row["latent_size"])
    # Create AutoEncoder instance dynamically
    hidden_sizes = list(map(int, hidden_sizes.split(', ')))

    return AutoEncoder(input_size=input_size, hidden_sizes=hidden_sizes, latent_size=int(latent_size))


# def train(net, trainloader, optim, epochs, device: str, mu, global_weights):
#     """Train the network on the training set."""
#     criterion = torch.nn.CrossEntropyLoss()
#     net.train()
#     for _ in range(epochs):
#         for batch in trainloader:
#             features, labels = batch[0].to(device), batch[1].to(device)
#             optim.zero_grad()
#             loss = criterion(net(features), labels)

#             ## fedprox code
#             prox_term = 0.0
#             for param, global_param in zip(net.parameters(), global_weights):
#                 prox_term += torch.norm(param - global_param, p=2) ** 2  # L2 Norm
#             loss += (mu / 2) * prox_term  # Add the FedProx penalty

#             loss.backward()
#             optim.step()

##Train Auto Encoder
def train(net, trainloader, optim, epochs, device: str, mu, global_weights):
    """Train the AutoEncoder on the training set with FedProx regularization."""
    criterion = torch.nn.MSELoss()  # Use MSE loss for reconstruction
    net.train()
    
    for _ in range(epochs):
        for batch in trainloader:
            features = batch[0].to(device)  # No labels needed for AutoEncoder
            optim.zero_grad()
            
            outputs = net(features)
            loss = criterion(outputs, features)  # Compare output to input
            
            # FedProx regularization term
            prox_term = 0.0
            for param, global_param in zip(net.parameters(), global_weights):
                prox_term += torch.norm(param - global_param, p=2) ** 2  # L2 Norm
            loss += (mu / 2) * prox_term  # Add the FedProx penalty
            
            loss.backward()
            optim.step()


# ##Evaluation
# def test(net, testloader, device: str):
#     """Validate the network on the entire test set."""
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, loss = 0, 0.0
#     net.eval()
#     with torch.no_grad():
#         for batch in testloader:
#             features, labels = batch[0].to(device), batch[1].to(device)
#             outputs = net(features)
#             loss += criterion(outputs, labels).item()
#             _, predicted = torch.max(outputs.data, 1)
#             correct += (predicted == labels).sum().item()
#     accuracy = correct / len(testloader.dataset)
#     return loss, accuracy

def test(net, testloader, device: str):
    """Validate the AutoEncoder on the entire test set."""
    criterion = torch.nn.MSELoss()  # Use MSE loss for reconstruction
    total_loss = 0.0
    net.eval()
    
    with torch.no_grad():
        for batch in testloader:
            features = batch[0].to(device)  # No labels needed for AutoEncoder
            outputs = net(features)
            loss = criterion(outputs, features)
            total_loss += loss.item() * features.size(0)
    
    avg_loss = total_loss / len(testloader.dataset)  # Average loss over all samples
    return avg_loss


    
    ##After each round it will be used
# def get_evaluate_fn(centralized_testset):
#     """This is a function that returns a function. The returned
#     function (i.e. `evaluate_fn`) will be executed by the strategy
#     at the end of each round to evaluate the stat of the global
#     model."""

#     def evaluate_fn(server_round: int, parameters, config):
#         """This function is executed by the strategy it will instantiate
#         a model and replace its parameters with those from the global model.
#         The, the model will be evaluate on the test set (recall this is the
#         whole MNIST test set)."""

#         model = construct_autoencoder(input_size=NUM_FEATURES)

#         # Determine device
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         model.to(device)  # send model to device

#         # set parameters to the model
#         params_dict = zip(model.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
#         model.load_state_dict(state_dict, strict=True)

#         # Save the model after the final round
#         if server_round == NUM_ROUNDS:  #NUM_ROUNDS is defined globally
#             torch.save(model.state_dict(), prepare_file_path(GLOBAL_MODEL_PATH))
#             print(f"Global model saved at round {server_round}")

#         # Apply transform to dataset
#         #testset = centralized_testset.with_transform(apply_transforms)
#         testloader = DataLoader(to_tensor(centralized_testset, "eval"), batch_size=BATCH_SIZE)
#         # call test
#         loss, accuracy = test(model, testloader, device)
#         return loss, {"accuracy": accuracy}

#     return evaluate_fn   

def get_evaluate_fn(centralized_testset):
    """Returns a function that evaluates the global AutoEncoder model."""
    
    def evaluate_fn(server_round: int, parameters, config):
        """Evaluate the global model at each round."""
        model = construct_autoencoder(input_size=NUM_FEATURES)
        
        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)  # Send model to device
        
        # Set parameters to the model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        # Save the model after the final round
        if server_round == NUM_ROUNDS:  # NUM_ROUNDS is defined globally
            torch.save(model.state_dict(), prepare_file_path(GLOBAL_MODEL_PATH))
            print(f"Global model saved at round {server_round}")
        
        # Prepare test loader
        testloader = DataLoader(to_tensor(centralized_testset, "eval"), batch_size=BATCH_SIZE)
        
        # Evaluate AutoEncoder
        loss = test(model, testloader, device)
        return loss, {"reconstruction_loss": loss}
    
    return evaluate_fn

##clear the cache of the Cuda
def clear_cuda_cache():
    torch.cuda.empty_cache()
    print("CUDA cache cleared.")


## Convert a panda dataframe into a Tensordataset for to be useable by torch
## @df = panda dataframe
def to_tensor(df, type="train"):
    """Convert DataFrame to PyTorch TensorDataset (Unsupervised - No Labels)."""
    if type == "eval":
        X = df.drop(columns=["Label"]).values  # Drop label column for input features
        y = df["Label"].values  # Extract labels
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)  # Ensure correct tensor type for labels
        return TensorDataset(X_tensor, y_tensor)
    else:
        X = df.values      
        X_tensor = torch.tensor(X, dtype=torch.float32)
        return TensorDataset(X_tensor)



## Prepare File Path from Features and Folds
def prepare_file_path(path):
    file_path = path.format(FOLDER_NAME.format(FEATURE_TYPE, NUM_FEATURES, FOLD))
    # Extract the directory path
    directory = os.path.dirname(file_path)
    # Check if the directory exists
    if not os.path.exists(directory):
        # If the directory does not exist, create it
        os.makedirs(directory, exist_ok=True)
    return file_path





