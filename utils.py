''' 
We'll be training the model in a Federated setting. In order to do that, we need to define two functions:

* `train()` that will train the model given a dataloader.
* `test()` that will be used to evaluate the performance of the model on held-out data, e.g., a training set.
'''
from config import NUM_ROUNDS, GLOBAL_MODEL_PATH, NUM_CLASSES, BATCH_SIZE, FOLDER_NAME, FOLD, NUM_FEATURES, FEATURE_TYPE,  LOCAL_TRAIN_HISTORY_PATH
from model import Net
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
import os
import csv
import timeit


def train(net, trainloader, optim, epochs, device: str):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            features, labels = batch[0].to(device), batch[1].to(device)
            optim.zero_grad()
            loss = criterion(net(features), labels)
            loss.backward()
            optim.step()


## This function will train the model with early stopping implemented

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_with_early_stopping(
    net,
    trainloader,
    testloader,
    optimizer,
    epochs,
    device: str,
    lr_patience=3,
    early_stop_patience=6,
    factor=0.1,  # Factor by which the LR will be reduced
    min_lr=0.00001,  # Minimum LR after reduction
    alpha=0.1 #to be used to calcuate penalty
):
    """Train the network with ReduceLROnPlateau scheduler, early stopping, and learning rate tracking."""
    criterion = torch.nn.CrossEntropyLoss()
    penalty = calculate_penalty(net, testloader, device, alpha)

    net.train() #put the model in trainign mode

    # Early stopping variables
    best_val_accuracy = 0
    epochs_without_improvement = 0
    best_model_state = None  # To save the best model

    # Store metrics for each epoch
    metrics_history = {
        "epoch": [],
        "training_loss": [],
        "training_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "learning_rate": [],  # To track learning rate for each epoch
        "training_time": [] #track the trainign time
    }

    # ReduceLROnPlateau Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=lr_patience, factor=factor, min_lr=min_lr)

    for epoch in range(epochs):
        #start the timer
        start_time = timeit.default_timer()

        # Training loop
        net.train()
        correct_train, total_train, train_loss = 0, 0, 0.0
        for iteration, batch in enumerate(trainloader):
            features, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss += penalty ##adding the penalty
            loss.backward()
            optimizer.step()

            # Calculate training metrics
            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        #end time
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        metrics_history["training_time"].append( elapsed) #tracking the time

        # Calculate epoch-wise training loss and accuracy
        epoch_train_loss = train_loss / total_train
        epoch_train_accuracy = correct_train / total_train
        metrics_history["training_loss"].append(epoch_train_loss)
        metrics_history["training_accuracy"].append(epoch_train_accuracy)

        # Validation loop
        val_loss, val_accuracy = test(net, testloader, device)
        metrics_history["validation_loss"].append(val_loss)
        metrics_history["validation_accuracy"].append(val_accuracy)

        # Track current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        metrics_history["learning_rate"].append(current_lr)

        #assigning current epoch
        metrics_history['epoch'].append(epoch + 1)

        #print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_train_loss:.4f}, Training Accuracy: {epoch_train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Learning Rate: {current_lr:.6f}")

        # Update the scheduler with the validation accuracy
        scheduler.step(val_accuracy)

        # Early stopping condition
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            best_model_state = net.state_dict()  # Save the best model state
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stop_patience: #early stop patience should be 2xpatience
            #print(f"Early Stopping Triggered at Epoch {epoch + 1}.")
            break

    # Return the best model state and metrics history
    return {
        "model_state": best_model_state if best_model_state is not None else net.state_dict(),
        "metrics_history": metrics_history,
    }


## calcuate penalty for Fasle Posivite Rate
## Calcuate Penaly
def calculate_penalty(net, testloader, device: str, alpha: float):
    criterion = torch.nn.CrossEntropyLoss()
    false_positives, false_negatives = 0, 0

    net.eval()    
    with torch.no_grad():
        for batch in testloader:
            features, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(features)           
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            # Calculate false positives and false negatives
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()
        
    return alpha * (false_negatives + false_positives) / len(testloader.dataset) #calculation of penalty



##Evaluate the function
def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            features, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(features)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

    
    ##After each round it will be used
def get_evaluate_fn(centralized_testset):
    """This is a function that returns a function. The returned
    function (i.e. `evaluate_fn`) will be executed by the strategy
    at the end of each round to evaluate the stat of the global
    model."""

    def evaluate_fn(server_round: int, parameters, config):
        """This function is executed by the strategy it will instantiate
        a model and replace its parameters with those from the global model.
        The, the model will be evaluate on the test set (recall this is the
        whole MNIST test set)."""

        model = Net(num_classes=NUM_CLASSES)

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)  # send model to device

        # set parameters to the model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Save the model after the final round
        if server_round == NUM_ROUNDS:  #NUM_ROUNDS is defined globally
            torch.save(model.state_dict(), prepare_file_path(GLOBAL_MODEL_PATH))
            print(f"Global model saved at round {server_round}")

        # Apply transform to dataset
        #testset = centralized_testset.with_transform(apply_transforms)
        testloader = DataLoader(to_tensor(centralized_testset), batch_size=BATCH_SIZE)
        # call test
        loss, accuracy = test(model, testloader, device)
        return loss, {"accuracy": accuracy}

    return evaluate_fn   

##clear the cache of the Cuda
def clear_cuda_cache():
    torch.cuda.empty_cache()
    print("CUDA cache cleared.")


## Convert a panda dataframe into a Tensordataset for to be useable by torch
## @df = panda dataframe
def to_tensor(df):
    # Separate features and labels
    X = df.drop(columns="Label").values  # Replace "label" with the actual label column name
    y = df["Label"].values
    # Convert to PyTorch tensors
    # Also reshaping it 5 by 6
    #X_tensor = torch.tensor(X.reshape(-1, 1, 5, 6), dtype=torch.float32)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X_tensor, y_tensor)


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


## Saving all the metrics to CSV
def save_metrics_to_csv(client_fit_metrics, client_eval_metrics, server_fit_metrics, server_eval_metrics, file_path):
   # Initialize CSV headers and rows
    headers = ["Client/Server", "Round", "Fit Accuracy", "Eval Accuracy"]
    rows = []

    # Process Client Fit and Eval Metrics
    all_rounds = set(client_fit_metrics['accuracy'].keys()).union(client_eval_metrics['accuracy'].keys())
    all_clients = set(client_id for round_data in client_fit_metrics['accuracy'].values() for client_id, _ in round_data)
    all_clients.update(client_id for round_data in client_eval_metrics['accuracy'].values() for client_id, _ in round_data)

    for round_num in sorted(all_rounds):
        for client_id in sorted(all_clients):
            fit_accuracy = next((acc for cid, acc in client_fit_metrics['accuracy'].get(round_num, []) if cid == client_id), None)
            eval_accuracy = next((acc for cid, acc in client_eval_metrics['accuracy'].get(round_num, []) if cid == client_id), None)
            if fit_accuracy is not None or eval_accuracy is not None:  # Only add if there's at least one metric
                rows.append([f"Client {client_id}", round_num, fit_accuracy, eval_accuracy])

    # Process Server Fit and Eval Metrics
    for round_num in sorted(server_fit_metrics['accuracy']):
        fit_accuracy = server_fit_metrics['accuracy'].get(round_num)
        eval_accuracy = server_eval_metrics['accuracy'].get(round_num, None)
        rows.append(["Server", round_num, fit_accuracy, eval_accuracy])
    
    # Write data to CSV
    with open(prepare_file_path(file_path), mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f" Metric Saved: {prepare_file_path(file_path)}")


## This function will save client wise local metrics
def save_local_train_history_to_csv(client_id, server_round, metrics):

    #print(server_round)

    #Prepare file path
    file_path = LOCAL_TRAIN_HISTORY_PATH.format("{}", client_id)

    # Initialize CSV headers and rows
    headers = ["Round", "Client", "Epoch", "Learing_Rate", "Train Accuracy", "Train Loss", "Validation Accuracy", "Validation Loss", "Training Time (S)"]
    rows = []

    # Check if the file exists
    if os.path.exists(prepare_file_path(file_path)):
        mode = "a"  # Append mode
    else:
        mode = "w"  # Write mode

    with open(prepare_file_path(prepare_file_path(file_path)), mode=mode, newline="") as file:
        writer = csv.writer(file)
        if mode == "w":
            writer.writerow(headers)

        for i, epoch in enumerate(metrics['epoch']):
            training_accuracy = metrics['training_accuracy'][i] if metrics['training_accuracy'][i] > 0 else 0
            validation_accuracy = metrics['validation_accuracy'][i] if metrics['validation_accuracy'][i] > 0 else 0
            training_loss = metrics['training_loss'][i] if metrics['training_loss'][i] > 0 else 0
            validation_loss = metrics['validation_loss'][i] if metrics['validation_loss'][i] > 0 else 0
            learning_rate = metrics['learning_rate'][i] if metrics['learning_rate'][i] > 0 else 0
            training_time = metrics['training_time'][i] if metrics['training_time'][i] > 0 else 0

            #Writing the row
            writer.writerow([server_round, client_id, epoch, learning_rate, training_accuracy, training_loss, validation_accuracy, validation_loss, training_time])

    #print(f"Local Training Metric Saved: {prepare_file_path(file_path)}")
   

## Save the mode based on parameters
def save_model(parameters, file_path = GLOBAL_MODEL_PATH, num_classes=NUM_CLASSES):
    try:
        model = Net(num_classes=num_classes)
        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)  # send model to device

        # set parameters to the model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        torch.save(model.state_dict(), prepare_file_path(file_path))
    except Exception as e:
        print(f"Saving model error: {e}")
