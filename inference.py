import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from dataloader import get_evaluation_datasets_by_client  # Assuming this function gets local client datasets
from collections import OrderedDict
from config import NUM_CLASSES, GLOBAL_MODEL_PATH, BATCH_SIZE, NUM_FEATURES
from torch.utils.data import DataLoader
from utils import to_tensor, construct_autoencoder
import pandas as pd
import time



STRATEGY = "Anomaly_FLAccShield"
ACCUMULATED_RESULTS = "./results/1.3_accumulated_results.csv"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the global model from the saved path
def load_model(input_size, model_path=GLOBAL_MODEL_PATH):

    # Ensure CUDA is not holding stale memory from previous models
    torch.cuda.empty_cache()  # Clears GPU memory cache
    model = construct_autoencoder(input_size=input_size)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.eval()
    return model

result_sources = {
    'components': range(27, 42),
    'folds': [1, 2, 3, 4, 5],
    'marker': ['o', '-', '^' 'x', '-o-'],
    'clients': [1, 2, 3, 4],
    'path': './results/original_{0}_fold_{1}'
}



client_metrics = {
    'Strategy': [],
    'Component': [],
    'Fold': [],
    'Client': [],
    'Accuracy': [],
    'F1_Score': [],
    'Precision': [],
    'Recall': [],
    'ROC_AUC': [],
    'Confusion_Matrix': [],
    'CM_String': [],
    'Sample_Number': [],
    'Inference_Time_Per_Sample': []
}

def run_unsupervised_inference(model, dataloader, device):
    """Run inference on AutoEncoder model for anomaly detection."""
    reconstruction_errors = []
    all_labels = []
    
    total_samples = len(dataloader.dataset)
    start_time = time.time()
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)            
            # Compute per-sample reconstruction error (MSE loss)
            errors = torch.mean((outputs - inputs) ** 2, dim=1).cpu().numpy()
            reconstruction_errors.extend(errors)
            all_labels.extend(labels.cpu().numpy())
    
    end_time = time.time()
    inference_time_per_sample = (end_time - start_time) * 1000000 / total_samples  # Microseconds per sample
    
    return np.array(reconstruction_errors), np.array(all_labels), f'{inference_time_per_sample:.4f} us'

def accumulate_results(results, confusion_matrix_data, model_to_use):
    components = results.get('components')
    folds = results.get('folds')
    path = results.get('path')
    clients = results.get('clients')
    
    for component in components:  
        for fold in folds:
            global_model = path.format(component, fold) + '/' + model_to_use            
            print(global_model)
            model = load_model(input_size=component, model_path=global_model)           
            
            for client in clients:                                
                testset = get_evaluation_datasets_by_client(client, fold=fold, feature_count=component) 
                testloader = DataLoader(to_tensor(testset, "eval"), batch_size=BATCH_SIZE)
                
                # Running inference
                reconstruction_errors, labels, inference_time_per_sample = run_unsupervised_inference(model, testloader, device)
                
                # Compute anomaly detection threshold (95th percentile of benign samples)
                benign_errors = [err for err, lbl in zip(reconstruction_errors, labels) if lbl == 0]
                #threshold = np.percentile(benign_errors, 95)  
                threshold = 0.2
                
                # Predict anomalies (1 if reconstruction error > threshold, else 0)
                preds = (np.array(reconstruction_errors) > threshold).astype(int)
                
                # Compute evaluation metrics
                avg_loss = np.mean(reconstruction_errors)
                accuracy = accuracy_score(labels, preds)
                f1 = f1_score(labels, preds)
                precision = precision_score(labels, preds)
                recall = recall_score(labels, preds)
                roc_auc = roc_auc_score(labels, reconstruction_errors)
                cm = confusion_matrix(labels, preds)
                cm_str = f"[[{cm[0][0]} {cm[0][1]}] [{cm[1][0]} {cm[1][1]}]]"
                
                # Store results
                client_metrics['Strategy'].append(STRATEGY)
                client_metrics['Component'].append(component)
                client_metrics['Fold'].append(fold)
                client_metrics['Client'].append(client)
                client_metrics['Accuracy'].append(accuracy)
                client_metrics['F1_Score'].append(f1)
                client_metrics['Precision'].append(precision)
                client_metrics['Recall'].append(recall)
                client_metrics['ROC_AUC'].append(roc_auc)
                client_metrics['Confusion_Matrix'].append(cm)
                client_metrics['CM_String'].append(cm_str)
                client_metrics['Sample_Number'].append(len(labels))
                client_metrics['Inference_Time_Per_Sample'].append(inference_time_per_sample)

                # Save confusion matrix data
                key = f'{component}_{fold}_{client}'
                confusion_matrix_data[key] = {
                    'preds': preds,
                    'labels': labels
                }   
        
            #delete the model and empty the cache
            del model
            torch.cuda.empty_cache()  # Clears all memory

    # Convert results to DataFrame
    df = pd.DataFrame(client_metrics)
    df.to_csv(ACCUMULATED_RESULTS, index=False)
    #print(df.to_string(index=False)
    return df, confusion_matrix_data   

if __name__ == "__main__":
    confusion_matrix_data = {}
    result_df, store_results_df = accumulate_results(result_sources, confusion_matrix_data, 'best_global_model.pth')
