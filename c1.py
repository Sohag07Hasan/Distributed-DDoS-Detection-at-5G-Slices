import flwr as fl
from dataloader import get_training_datasets_by_client
from client import create_client
from config import SERVER_ADDRESS

#client number
client_id = 1


if __name__ =="__main__":
    ## Collecting Datasets
    training, validation = get_training_datasets_by_client(client_id=client_id)
    #print(training.shape)
    #print(training.columns)
  
    #fl.client.start_numpy_client(
    fl.client.start_client(
        server_address=SERVER_ADDRESS, 
        client=create_client(training, validation, client_id)
    )
