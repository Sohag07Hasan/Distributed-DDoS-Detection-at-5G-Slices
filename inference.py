import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from dataloader import get_evaluation_datasets_by_client  # Assuming this function gets local client datasets
from model import Net
from collections import OrderedDict
from config import NUM_CLASSES, NUM_CLIENTS, GLOBAL_MODEL_PATH, BATCH_SIZE, NUM_FEATURES
from torch.utils.data import DataLoader
from utils import to_tensor
import pandas as pd
import pickle
import time
import os

STRATEGY = "FedProx"


# Load the global model from the saved path
def load_model(model_path=GLOBAL_MODEL_PATH, input_size=NUM_FEATURES, num_classes=NUM_CLASSES):
    if os.path.exists(model_path):
        model = Net(input_size=input_size, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    else:
        return False

# Run inference on a client's dataset
def run_inference(model, dataloader, device):
    all_preds = []
    all_labels = []

    # Get the total number of samples from the DataLoader
    total_samples = len(dataloader.dataset)
    # Start the timer before the loop
    start_time = time.time()
    
    with torch.no_grad():
        for batch in dataloader:
            features, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(features)
            _, preds = torch.max(outputs.data, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # End the timer after the loop
    end_time = time.time()    
    # Calculate the total inference time
    total_inference_time = end_time - start_time    
    # Calculate average inference time per sample
    #inference_time_per_sample =  total_inference_time * 1000
    inference_time_per_sample =  total_inference_time * 1000000 / total_samples
    
    return np.array(all_preds), np.array(all_labels), f'{inference_time_per_sample:.4f} us'




def accumulate_results(results, confusion_matrix_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    client_metrics = {
        'Strategy': [],
        'Component': [],
        'Fold': [],
        'Client': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1_Score': [],
        'Sample_Number': [],
        'Inference_Time_Per_Sample': []
    }

    components = results.get('components')
    folds = results.get('folds')
    path = results.get('path')
    clients = results.get('clients')
     
    for component in components:  
        for fold in folds:
            
            global_model = path.format(component, fold) + '/' + 'global_model.pth'
            history = path.format(component, fold) + '/' + 'history.pkl'
            #training_time = path.format(component, fold) + '/' + 'training_time.txt'
            
            model = load_model(model_path=global_model, input_size=component, num_classes=2)
            
            if model == False:
                continue

            model.to(device)   

            print(f"inferecne running: Feature {component}, Fold {fold}")
            
            for client in clients:                                
                testset = get_evaluation_datasets_by_client(client, fold=fold, feature_count=component)  
                testloader = DataLoader(to_tensor(testset), batch_size=BATCH_SIZE)
                preds, labels, inference_time_per_sample = run_inference(model, testloader, device)
                client_metrics['Strategy'].append(STRATEGY)
                client_metrics['Component'].append(component)
                client_metrics['Fold'].append(fold)
                client_metrics['Client'].append(client)
                client_metrics['Accuracy'].append(accuracy_score(labels, preds))
                client_metrics['Precision'].append(precision_score(labels, preds, zero_division=1))
                client_metrics['Recall'].append(recall_score(labels, preds, zero_division=1))
                client_metrics['F1_Score'].append(f1_score(labels, preds, zero_division=1))
                client_metrics['Sample_Number'].append(len(labels)),
                client_metrics['Inference_Time_Per_Sample'].append(inference_time_per_sample)

                #Saving info for confusion matrix
                key = f'{component}_{fold}_{client}'
                confusion_matrix_data[key] = {
                    'preds': preds,
                    'labels': labels,
                    'classes': np.arange(NUM_CLASSES)
                }   

    ##Converting into datafram for better visualization
    df = pd.DataFrame(client_metrics)
    #print(df.to_string(index=False))
    return df, confusion_matrix_data           
    


if __name__ == "__main__":
    result_sources = {
        'components': [4, 6, 8, 10, 12],
        'folds': [1, 2, 3, 4, 5],
        #'folds': [1, 2],
        'marker': ['o', '-', '^' 'x', '-o-'],
        'clients': [1, 2, 3, 4, 5, 6],
        'path': './results/original_{0}_fold_{1}'
    }
    confusion_matrix_data = {}
    result_df, store_results_df = accumulate_results(result_sources, confusion_matrix_data)
    result_df.to_csv("./results/6_client_results.csv", index=False)
