from sklearn.metrics import classification_report
import logging
import torch
import pickle
import os
import cloudpickle

def calc_metrics(y_pred,y_true,additional_metrics:dict=None):
    metrics_dict = classification_report(
        y_true, 
        y_pred, 
        output_dict=True,
        zero_division=0
        )
    if additional_metrics:
        metrics_dict.update(additional_metrics)

    for key,value in metrics_dict.items():
        if key in ["accuracy","loss_total"]:
            print(f"{key}:{value}")

def start_logging(experiment_name='test'):
    logger = logging.getLogger(experiment_name)
    return logger

def save_model(model, model_dir, run_id):

    model_path=os.path.join(model_dir, run_id+'.pkl')
    with open(model_path, 'wb') as f:
        cloudpickle.dump(model, f)

def load_model(model_trained):

    model_path=model_trained
    with open(model_path, 'rb') as f:
        model = cloudpickle.load(f)

    return model