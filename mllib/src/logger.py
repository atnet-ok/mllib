from sklearn.metrics import classification_report
import logging
import torch
import pickle
import os
from cloudpickle import dump, load
from mllib.src.config import *
# from joblib import dump, load
# from pickle import dump, load

def start_experiment(args):
    config_path=os.path.join(args.cfg_dir, args.run_id+'.yaml')
    logger = start_logging(args.experiment_name)
    cfg = get_config(config_path=config_path)
    return cfg, logger

def end_experiment(args, logger, model, metrics):
    if model:
        save_model(model, args.model_dir, args.run_id)



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
        dump(model, f)

def load_model(model_path):

    with open(model_path, 'rb') as f:
        model = load(f)
    return model