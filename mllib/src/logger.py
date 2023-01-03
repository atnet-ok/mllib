from sklearn.metrics import classification_report
from dataclasses import  asdict
from cloudpickle import dump, load
from mllib.src.config import *
# from joblib import dump, load
# from pickle import dump, load
import logging
import os

class MyLogger():
    def __init__(self, experiment_name, log_path) -> None:
        self.logger =  logging.getLogger(experiment_name)
        self.logger.setLevel(logging.DEBUG)
        sh = logging.StreamHandler()
        self.logger.addHandler(sh)
        fh = logging.FileHandler(log_path)
        self.logger.addHandler(fh)

    def log(self, message):
        if type(message)==config:
            message = asdict(message)
        self.logger.info(message)

    def log_metrics(self, y_pred, y_true, additional_metrics:dict=None):
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
                self.log(f"{key}:{value}")

        return metrics_dict

def start_experiment(args):
    log_path = os.path.join(args.log_dir, args.run_id+'.log')
    logger = MyLogger(args.experiment_name, log_path)

    config_path=os.path.join(args.cfg_dir, args.run_id+'.yaml')
    cfg = get_config(config_path=config_path)

    logger.log('Now Starting '+ args.experiment_name)
    logger.log('run id is '+ args.run_id)
    logger.log(cfg)
    return cfg, logger

def end_experiment(args,  model, metrics):
    if model:
        save_model(model, args.model_dir, args.run_id)
    

def save_model(model, model_dir, run_id):

    model_path=os.path.join(model_dir, run_id+'.pkl')
    with open(model_path, 'wb') as f:
        dump(model, f)

def load_model(model_path):

    with open(model_path, 'rb') as f:
        model = load(f)
    return model