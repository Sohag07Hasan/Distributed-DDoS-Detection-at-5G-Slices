import flwr as fl
from config import (
    FRACTION_FIT, FRACTION_EVAL, NUM_CLIENTS, MIN_FIT_CLIENTS, 
    MIN_EVAL_CLIENTS, Q_PARAM
)
from typing import Callable


def create_strategy(fit_config_fn: Callable, weighted_average_fn: Callable, evaluate_fn: Callable, fit_metrics_aggregation_fn: Callable):
    strategy = fl.server.strategy.QFedAvg(
        fraction_fit = FRACTION_FIT,  # The fraction of clients used for training in each round
        fraction_evaluate = FRACTION_EVAL,  # The fraction of clients used for evaluation in each round.
        min_fit_clients = MIN_FIT_CLIENTS,  # This ensures that at least NUM_CLIENTS must be available and must successfully complete training in each round. If fewer than this number of clients are available or successfully complete training, the round will fail
        min_evaluate_clients = MIN_EVAL_CLIENTS,  # Minimum number of clients for evaluation
        min_available_clients = NUM_CLIENTS,  # The minimum number of clients that must be available in the system for a round to start.
        on_fit_config_fn = fit_config_fn, #A callback function that returns a dictionary of configurations that will be passed to each client before training starts
        #fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,  # A function to aggregate the metrics from all clients after the training round
        evaluate_metrics_aggregation_fn = weighted_average_fn,  # A function to aggregate the evaluation metrics from all clients after the evaluation round.
        evaluate_fn = evaluate_fn,  # A function to evaluate the global model on the server side after each round
        q_param=Q_PARAM
    )
    return strategy
