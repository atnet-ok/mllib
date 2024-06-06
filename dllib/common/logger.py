from dllib.config import *
import mlflow
from omegaconf import OmegaConf
from dataclasses import dataclass, field, asdict


class Logger:
    def __init__(self, logger_cfg: logger_cfg = logger_cfg()) -> None:
        mlflow.set_tracking_uri(logger_cfg.log_uri)
        mlflow.set_experiment(experiment_name=logger_cfg.experiment_name)

        with mlflow.start_run(run_name=logger_cfg.run_name) as run:
            self.run_id = run.info.run_id

    def log_metrics(self, metrics: dict, step: int):
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_metrics(metrics=metrics, step=step)

    def log_params(self, params: dict):
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_params(params=params)

    def log_config(self, config: dict):
        def log_config_recursive(dct, prex=""):
            for key, value in dct.items():
                if type(value) == dict:
                    prex = str(key) + "/"
                    log_config_recursive(value, prex)
                else:
                    key = prex + str(key)

                    with mlflow.start_run(run_id=self.run_id):
                        mlflow.log_param(key=key, value=value)

        log_config_recursive(config, prex="")

    def log_artifacts(self, file_path):
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_artifacts(local_dir=file_path)

    def log_model(self, model, model_name):
        with mlflow.start_run(run_id=self.run_id):
            mlflow.pytorch.log_model(model, model_name)
