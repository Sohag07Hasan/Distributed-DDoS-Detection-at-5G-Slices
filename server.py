import flwr as fl
from strategy import create_strategy  
from dataloader import get_centralized_testset
from config import LEARNING_RATE, EPOCHS, NUM_ROUNDS, SERVER_ADDRESS, NUM_ROUNDS, HISTORY_PATH_TXT, HISTORY_PATH_PKL, TRAINING_TIME, EARLY_STOPPING_ROUNDS, IMPROVEMENT_THRESHOLD
from flwr.common import Metrics, Scalar
from utils import get_evaluate_fn, clear_cuda_cache, prepare_file_path
from typing import Dict, List, Tuple
#import matplotlib.pyplot as plt
import pickle
import time
import os


## Collecting Datasets
centralized_testset = get_centralized_testset()

# Early stopping parameters
early_stopping_rounds = EARLY_STOPPING_ROUNDS  # Stop training if no significant improvement after this many rounds
improvement_threshold = IMPROVEMENT_THRESHOLD  # Minimum required improvement in accuracy to continue
best_accuracy = 0.0
no_improvement_counter = 0  # Counter to track rounds with no improvement


#before fitting each client locally this function will send the fit configuraiton
def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": EPOCHS,  # Number of local epochs done by clients
        "lr": LEARNING_RATE,  # Learning rate to use by clients during fit()
        "server_round": server_round ##Number of Rounds sent to Client
    }
    return config

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"loss": sum(losses) / sum(examples)}


def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Example: compute weighted average accuracy
    #print(metrics)
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"loss": sum(losses) / sum(examples)}

#Store the history
def save_history(history, path_text=HISTORY_PATH_TXT, path_pkl=HISTORY_PATH_PKL):
    with open(prepare_file_path(path_pkl), "wb") as file:
        pickle.dump(history, file)

        # Save as a plain text file
    with open(prepare_file_path(path_text), 'w') as file:
        file.write(str(history))
    print(f"history saved as text @ {prepare_file_path(path_text)}")
    print(f"history saved as picle @ {prepare_file_path(path_pkl)}")
    
#Store Training time 
def save_training_time(start_time, end_time):
    training_time = end_time - start_time
           # Save as a plain text file
    with open(prepare_file_path(TRAINING_TIME), 'w') as file:
        file.write(str(training_time))


if __name__ == "__main__":
    
    #clear the GPU cache
    clear_cuda_cache()

    # Start timing
    start_time = time.time()

    # Define the server configuration and start the server
    server_config = fl.server.ServerConfig(num_rounds=NUM_ROUNDS)
    strategy = create_strategy(fit_config, weighted_average, get_evaluate_fn(centralized_testset), fit_metrics_aggregation)
    
    history = fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=server_config,
        strategy=strategy,  # Use the imported strategy
    )
    # End timing
    end_time = time.time()

    save_training_time(start_time, end_time)
    save_history(history)

     # Write "done" to a file when server finishes training
    with open("server_done.txt", "w") as f:
        f.write("done")

    print("Server training complete and flag written.")

