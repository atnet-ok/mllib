from cloudpickle import dump, load
from mllib.src.config import *
# from joblib import dump, load
# from pickle import dump, load
import logging
import mlflow

class MllibLogger():
    def __init__(self, experiment_name, log_path, model_path) -> None:


        print(log_path)
        self.model_path = model_path
        self.logger =  logging.getLogger(experiment_name)
        sh = logging.StreamHandler()
        fh = logging.FileHandler(log_path)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)
        self.logger.setLevel(logging.DEBUG)

    def log(self, message):
        self.logger.info(message)

    def log_metrics(self, metrics_dict, phase='train',additional_metrics:dict=None,step=None):


        if additional_metrics:
            metrics_dict.update(additional_metrics)

        for key,value in metrics_dict.items():
            key = key + '_' + phase
            # if ('accuracy' in key) or ('loss' in key):
            #     self.log(f"{key}:{value}")

            mlflow.log_metric(key=key,value=value,step=step)

        return metrics_dict

    def save_model(self, model):

        with open(self.model_path, 'wb') as f:
            dump(model, f)
        mlflow.log_artifact(self.model_path)

    def load_model(self, model_path):
        mlflow.log_artifact(model_path)
        with open(model_path, 'rb') as f:
            model = load(f)
        return model

