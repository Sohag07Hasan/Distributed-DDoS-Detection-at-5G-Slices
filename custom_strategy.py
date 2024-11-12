from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple, Callable, Optional, Union
from logging import WARNING

##configuration
from config import NUM_ROUNDS, BATCH_SIZE, GLOBAL_MODEL_PATH, NUM_CLASSES, METRIC_PATH, BEST_GLOBAL_MODEL_PATH
from dataloader import get_centralized_testset
from torch.utils.data import DataLoader, TensorDataset
from model import Net
from utils import to_tensor, test, prepare_file_path, save_metrics_to_csv, save_model
import torch
import numpy as np
from collections import OrderedDict
from functools import reduce
from typing import Callable

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Metrics
)

from flwr.common.logger import log
#from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


class CustomFedAvgEarlyStop(FedAvg):
    def __init__(
        self,
        initial_lr=0.01,
        initial_epochs=5,
        lr_adjustment_factor=0.1,
        min_lr=0.0001,
        improvement_threshold=0.01,  # Minimum accuracy improvement threshold
        max_rounds=20,               # Maximum number of rounds if not stopped early
        ff=1,
        fe=1,
        mfc=2,
        mec=2,
        mac=4
    ):
        super().__init__(fraction_fit=ff, fraction_evaluate=fe, min_fit_clients=mfc, min_evaluate_clients=mec, min_available_clients=mac, fit_metrics_aggregation_fn=weighted_average_fit, evaluate_metrics_aggregation_fn=weighted_average_eval, evaluate_fn=evaluate_fn)  # Initialize FedAvg with additional parameters
        self.initial_lr = initial_lr
        self.initial_epochs = initial_epochs
        self.lr_adjustment_factor = lr_adjustment_factor # not in use
        self.min_lr = min_lr # not in use
        self.fit_configs = {}  # Store individual client configurations
        self.client_fit_metrics = {'accuracy': {}, 'loss': {}}  # Track evaluation metrics per client
        self.client_eval_metrics = {'accuracy': {}, 'loss': {}}  # Track evaluation metrics per client
        self.server_fit_metrics = {'accuracy': {}, 'loss': {}}  # Track evaluation metrics for server
        self.server_eval_metrics = {'accuracy': {}, 'loss': {}}  # Track evaluation metrics for server
        self.global_models = {} ##Store global model parameters
        self.improvement_threshold = improvement_threshold ## accuracy should be improved by this threshold
        self.threshold_rounds_for_early_stopping = 5 
        self.max_rounds = max_rounds
        self.current_round = 0
        self.client_mapping = {}

        ##Early Stopping critera
        self.is_early_stop_applicable = False
        self.early_stopping_round = None  # Round at which early stopping occurred
        self.best_accuracies = {}  # Track best accuracy per client
        self.no_improvement_rounds = {}  # Track consecutive no-improvement rounds per client


    #Begining of a round 
    #Setup fit  parameters (like Learning Rate, Epoch etc.)
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple["ClientProxy", FitIns]]:
        """Configure the next round of training with client-specific LR and epochs."""
        
        self.current_round = server_round # storing current round

        fit_configurations = []
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        
        for client in clients:
            # Initialize client-specific configurations if not yet set
            if client.cid not in self.fit_configs:
                self.fit_configs[client.cid] = {
                    "lr": self.initial_lr,
                    "epochs": self.initial_epochs,
                }
            
            # Use the dynamically adjusted values for each client
            client_config = self.fit_configs[client.cid]
            fit_ins = FitIns(parameters, client_config)
            fit_configurations.append((client, fit_ins))
        
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average, incorporating normalized accuracy."""
        
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # Step 1: grabbing all the parameters
        accuracies = [fit_res.metrics.get("accuracy", 0.5) for _, fit_res in results]
        losses = [fit_res.metrics.get("loss", 0.0) for _, fit_res in results]
        num_examples = [fit_res.num_examples for _, fit_res in results]
        actual_client_ids = [fit_res.metrics.get("client_id", 0) for _, fit_res in results]
        fed_client_ids = [client.cid for client, _ in results]

        ## Storing client wise fit metrics
        self.client_fit_metrics['accuracy'][server_round] = list(zip(actual_client_ids, accuracies))
        self.client_fit_metrics['loss'][server_round] = list(zip(actual_client_ids, losses))

        if not self.client_mapping:
            self.client_mapping = dict(zip(actual_client_ids, fed_client_ids)) #Saving the mapping for future use

        # Step 2 Clients that are not generating good accuracy we will give their weights high priority
        # So we invese the weighted accuracy for that
        # total_examples = sum(num_examples)
        # print(f'Round: {server_round}; Total_examples: {total_examples}')
        # weighted_avg = [(accuracies[i] * num_examples[i]) / total_examples for i in range(len(accuracies))]
        # inverse_weighted_avg = [1 / value for value in weighted_avg]
        # print(inverse_weighted_avg)
        #total_inverse_weighted_avg = sum(inverse_weighted_avg)
        #new_weights = [value / total_inverse_weighted_avg for value in inverse_weighted_avg]

        inverse_accuracies = [1 / value for value in accuracies] #Inversing to give priority to the low performing clients

       # Step 3: Integrate new weights in weights
        # weights_results = [
        #     (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        #     for _, fit_res in results
        # ]

        weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        # Ensure all weights are on CPU as numpy arrays for aggregation
        # weights = [[tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in layer] for layer in weights]
        weighted_num_examples = np.multiply(num_examples, inverse_accuracies).tolist()

        aggregated_ndarrays = self.original_aggregate(list(zip(weights,  weighted_num_examples)))
        # print(f'Lenght of original: {len(aggregated_ndarrays)}')    
        # weighted_aggregated_ndarrays = self.aggregate(list(zip( aggregated_ndarrays, inverse_weighted_avg)))
        # print(f'Lenght of weighted: {len(weighted_aggregated_ndarrays)}')
        
        ## store the global model for each rounds
        self.global_models[server_round] = aggregated_ndarrays

        #converting back to parameters
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if an aggregation function is provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        #Storing Server info
        self.server_fit_metrics['accuracy'][server_round] = metrics_aggregated['accuracy']
        self.server_fit_metrics['loss'][server_round] = metrics_aggregated['loss']

        # returing aggregated parameters and metrics
        return parameters_aggregated, metrics_aggregated


    ## aggregate weights based on accuracy (new weights)
    # Original aggregate only works with integer. Now it will work for float as well
    def original_aggregate(self, results: list[tuple[np.ndarray, float]]) -> list[np.ndarray]:
        """Compute weighted average."""
        # Calculate the total number of examples (now allowing floats)
        num_examples_total = sum(num_examples for (_, num_examples) in results)

        # Create a list of weights, each multiplied by the related number of examples (as floats)
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

        # Compute average weights of each layer
        weights_prime = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime
    

    def aggregate(self, results: list[tuple[np.ndarray, float]]) -> list[np.ndarray]:
        """Compute weighted average of model weights from multiple clients."""
        # Calculate the total number of examples (as floats for accuracy)
        num_examples_total = sum(num_examples for _, num_examples in results)

        # Initialize weighted sum for each layer with zeros based on shape of first client weights
        weighted_sum = [np.zeros_like(layer) for layer in results[0][0]]

        # Compute weighted sum for each layer
        for weights, num_examples in results:
            for i, layer in enumerate(weights):
                weighted_sum[i] += layer * num_examples

        # Calculate the weighted average by dividing by the total number of examples
        weights_prime = [layer / num_examples_total for layer in weighted_sum]
        return weights_prime


    def should_continue_training(self) -> bool:
        """Check if training should continue."""
        # Stop if the current round exceeds max_rounds due to early stopping
        return self.current_round <= self.max_rounds and not self.is_early_stop_applicable

    def __repr__(self):
        return "CustomFedAvgEarlyStop with dynamic LR/epoch adjustment and early stopping"
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses and metrics using weighted average, with client-based early stopping."""
        
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        ## Grabbing metrics together
        accuracies = [eval_res.metrics.get("accuracy", 0.5) for _, eval_res in results]
        losses = [eval_res.metrics.get("loss", 0.0) for _, eval_res in results]
        actual_client_ids = [eval_res.metrics.get("client_id", 0) for _, eval_res in results]
        self.client_eval_metrics['accuracy'][server_round] = list(zip(actual_client_ids, accuracies))
        self.client_eval_metrics['loss'][server_round] = list(zip(actual_client_ids, losses))


        # Step 1: Aggregate loss with weighted average
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Step 2: Track client-specific accuracy improvements
        num_clients_with_no_improvement = 0  # Count of clients with no improvement in this round

        for client_proxy, evaluate_res in results:
            client_id = evaluate_res.metrics.get("client_id") #original client ID
            accuracy = evaluate_res.metrics.get("accuracy", 0.5) # Evaluation results

            # Initialize client's best accuracy and no-improvement counter if not already tracked
            if client_id not in self.best_accuracies:
                self.best_accuracies[client_id] = accuracy
                self.no_improvement_rounds[client_id] = 0
            elif accuracy > self.best_accuracies[client_id] + self.improvement_threshold:
                # Update best accuracy and reset no-improvement counter
                self.best_accuracies[client_id] = accuracy
                self.no_improvement_rounds[client_id] = 0
            else:
                # Increment no-improvement counter if no significant improvement
                self.no_improvement_rounds[client_id] += 1

            # Track clients with no improvement
            if self.no_improvement_rounds[client_id] >= self.threshold_rounds_for_early_stopping:
                num_clients_with_no_improvement += 1

        # Step 3: Check if early stopping conditions are met (more than 50% clients show no improvement)
        if num_clients_with_no_improvement > len(results) / 2:
            self.is_early_stop_applicable = True
            self.early_stopping_round = server_round
            log(WARNING, f"Early stopping triggered at round {server_round} due to no improvement in >50% of clients")

        # Aggregate custom metrics if aggregation function is provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")


        #Storing Server info in evaluation
        self.server_eval_metrics['accuracy'][server_round] = metrics_aggregated['accuracy']
        self.server_eval_metrics['loss'][server_round] = metrics_aggregated['loss']

        ##if it is last round is it save the metrics
        if server_round == self.max_rounds:
            self.save_all_data()

        return loss_aggregated, metrics_aggregated


    #Save info
    def save_all_data(self):

        #Save the last model
        save_model(self.global_models.get(self.current_round), file_path=GLOBAL_MODEL_PATH)

        ## Save the best model
        highest_accuracy_round = max(self.server_eval_metrics['accuracy'], key=self.server_eval_metrics['accuracy'].get)
        save_model(self.global_models.get(highest_accuracy_round), file_path=BEST_GLOBAL_MODEL_PATH)

        #Save the metrics
        save_metrics_to_csv(self.client_fit_metrics, self.client_eval_metrics, self.server_fit_metrics, self.server_eval_metrics, METRIC_PATH)



## Aggregate validation Accuracy
def weighted_average_eval(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}


## Aggregate Training Accuracy
def weighted_average_fit(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Custom aggregation function for training (fit) metrics."""
    # Calculate weighted accuracy based on number of examples each client used
    accuracies = [num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return the custom metric as weighted average
    if sum(examples) == 0:
        return {"accuracy": 0.0}  # Handle division by zero if no examples are used
    return {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}


## Centralize Evaluation if central Evaluation dataset available
## We accumulate all the Evaluation dataset for all the clients
## We do not make any decission based on this metrics. Only cross check with distributed evaluation
## Another point, we save global model when it reaches to the max round
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

    # # Save the model after the final round
    # if server_round == NUM_ROUNDS:  #NUM_ROUNDS is defined globally
    #     torch.save(model.state_dict(), prepare_file_path(GLOBAL_MODEL_PATH))
    #     print(f"Global model saved at round {server_round}")

    # collecting central testset
    centralized_testset = get_centralized_testset()

    testloader = DataLoader(to_tensor(centralized_testset), batch_size=BATCH_SIZE)
    # call test
    loss, accuracy = test(model, testloader, device)
    return loss, {"accuracy": accuracy}








