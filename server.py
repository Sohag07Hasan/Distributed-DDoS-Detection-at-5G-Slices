import flwr
from flwr.server.client_manager import SimpleClientManager
from flwr.common import Metrics, Scalar

# Standard Libraries
from concurrent import futures
import time
import pickle
import os
import timeit

# Custom Modules
from strategy_acc_shield import FLAccShield  # Your custom strategy
from server_acc_shield import ServerAccShield  # Custom server class
from utils import (
    clear_cuda_cache,
    prepare_file_path,
    get_evaluate_fn,
)  # Avoid repetitive utils imports
from config import (
    LEARNING_RATE,
    EPOCHS,
    NUM_ROUNDS,
    SERVER_ADDRESS,
    HISTORY_PATH_TXT,
    HISTORY_PATH_PKL,
    TRAINING_TIME,
    NUM_CLIENTS,
    FRACTION_FIT,
    FRACTION_EVAL,
    MIN_FIT_CLIENTS,
    MIN_EVAL_CLIENTS,
    LR_ADJUSTMENT_FACTOR,
    MIN_LR,
    #PATIENCE_ON_EPOCH,
    LR_ADJUSTMENT_PATIENCE_ON_EPOCH,
    EARLY_STOP_PATIENCE_ON_EPOCH,
    EARLY_STOP_PATIENCE_ON_ROUND,
    ROUND_ON_ROUND_ACCURACY_THRESHOLD,
    #Penalty Term to calcuate from False Positive Rate
    ALPHA
)
from dataloader import get_centralized_testset  # Centralized testset loader

# Typing
from typing import Dict, List, Tuple


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
def save_training_time(elapse):
    with open(prepare_file_path(TRAINING_TIME), 'w') as file:
        file.write(str(elapse))


def create_strategy():
    # Early stopping function
    strategy = FLAccShield(
        initial_lr = LEARNING_RATE,
        initial_epochs = EPOCHS,
        lr_adjustment_factor = LR_ADJUSTMENT_FACTOR,
        min_lr = MIN_LR,
        #improvement_threshold = 0.01,
        training_rounds = NUM_ROUNDS,
        lr_adjustment_patience_on_epoch = LR_ADJUSTMENT_PATIENCE_ON_EPOCH,
        early_stop_patience_on_epoch = EARLY_STOP_PATIENCE_ON_EPOCH,
        early_stop_patience_on_round = EARLY_STOP_PATIENCE_ON_ROUND,
        round_on_round_accuracy_threshold = ROUND_ON_ROUND_ACCURACY_THRESHOLD,
        ff = FRACTION_FIT, #fraction_fit
        fe = FRACTION_EVAL, #fraction_evaluate
        mfc = MIN_FIT_CLIENTS, # min_fit_clients
        mec = MIN_EVAL_CLIENTS, #min_evaluate_clients
        mac = NUM_CLIENTS, #min_available_clients
        alpha = ALPHA, #penalty term to use with fcr
    )

    return strategy


if __name__ == "__main__":
        #clear the GPU cache
    clear_cuda_cache()

    # Start timing
    start_time = timeit.default_timer()

    # Define client manager and custom strategy
    client_manager = SimpleClientManager()
    strategy = create_strategy()  # Your strategy

    # Instantiate the EarlyStoppingServer
    server = ServerAccShield(client_manager=client_manager, strategy=strategy)
    server_config = flwr.server.ServerConfig(num_rounds=NUM_ROUNDS)  

    history = flwr.server.start_server(
        server = server,
        server_address=SERVER_ADDRESS,
        config = server_config
    )

    # End timing
    end_time = timeit.default_timer()
    elapse = end_time - start_time  
    save_training_time(elapse)  #save training time

    #save history
    save_history(history)

    print("History:", history)

    # Write "done" to a file when server finishes training
    # Bash Script will understand and rerun the next fold
    with open("server_done.txt", "w") as f:
        f.write("done")
    print("Server training complete and flag written.")
    