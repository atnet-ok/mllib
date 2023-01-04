from sklearn.metrics import classification_report, accuracy_score, f1_score
from dataclasses import  asdict
from cloudpickle import dump, load
from mllib.src.config import *
# from joblib import dump, load
# from pickle import dump, load
import logging
import os
import mlflow

class MyLogger():
    def __init__(self, experiment_name, log_path) -> None:

        print(log_path)
        self.logger =  logging.getLogger(experiment_name)
        sh = logging.StreamHandler()
        fh = logging.FileHandler(log_path)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)
        self.logger.setLevel(logging.DEBUG)

    def log(self, message):
        self.logger.info(message)

    def log_metrics(self, y_pred, y_true, phase='train',additional_metrics:dict=None,step=None):

        metrics_dict = classification_report(
            y_true, 
            y_pred, 
            output_dict=True,
            zero_division=0
            )

        metrics_dict = summary_report(metrics_dict)

        if additional_metrics:
            metrics_dict.update(additional_metrics)

        for key,value in metrics_dict.items():
            key = key + '_' + phase
            if ('accuracy' in key) or ('loss' in key):
                self.log(f"{key}:{value}")

            mlflow.log_metric(key=key,value=value,step=step)

        return metrics_dict

def summary_report(metrics_dict:dict, save_acc_per_class=None):
    summary = dict()
    for key,value in metrics_dict.items():

        if key =='accuracy':
            summary.update(
                {key:value}
            )
        elif key =='macro avg':
            for key_t,value_t in value.items():
                summary.update(
                    {key_t:value_t}
                )      
        elif key =='weighted avg':
            pass
        else:
            if save_acc_per_class:
                for key_t,value_t in value.items():
                    if key_t=='f1-score':
                        summary.update(
                            {key+'_'+key_t:value_t}
                        )      
            else:
                pass 
        
    return summary


def start_experiment(args):
    log_path = os.path.join(args.log_dir, args.run_name+'.log')
    logger = MyLogger(args.experiment_name, log_path)

    config_path=os.path.join(args.cfg_dir, args.run_name+'.yaml')
    cfg = get_config(config_path=config_path)

    logger.log('-'*30)
    logger.log('Now Starting '+ args.experiment_name)
    logger.log('run id is '+ args.run_name)
    logger.log(cfg)

    mlflow.set_experiment(args.experiment_name)
    mlflow.start_run(run_name=args.run_name)
    dct = asdict(cfg)
    for key_0 in dct:
        for key, value in dct[key_0].items():
            mlflow.log_param(key_0+'_'+key, value)

    return cfg, logger

def end_experiment(args,  model, metrics):
    if model:
        save_model(model, args.model_dir, args.run_name)

    log_path = os.path.join(args.log_dir, args.run_name+'.log')
    config_path=os.path.join(args.cfg_dir, args.run_name+'.yaml')
    mlflow.log_artifact(log_path)
    mlflow.log_artifact(config_path)
    mlflow.end_run()
    

def save_model(model, model_dir, run_name):

    model_path=os.path.join(model_dir, run_name+'.pkl')
    with open(model_path, 'wb') as f:
        dump(model, f)
    mlflow.log_artifact(model_path)

def load_model(model_path):
    mlflow.log_artifact(model_path)
    with open(model_path, 'rb') as f:
        model = load(f)
    return model
