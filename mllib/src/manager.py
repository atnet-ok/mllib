import os
import mlflow
from mllib.src.logger import MllibLogger
from mllib.src.config import *
from mllib.src.trainer import *

class Manager():
    def __init__(self,args) -> None:
        self.args = args
        self.config_path = os.path.join(args.cfg_dir, args.run_name+'.yaml')
        self.log_path = os.path.join(args.log_dir, args.run_name+'.log')
        self.model_path = os.path.join(args.model_dir, args.run_name+'.pkl')

    def set_experiment(self):
        logger = MllibLogger(self.args.experiment_name, self.log_path, self.model_path)
        cfg = get_config(self.config_path)

        logger.log('-'*30)
        logger.log('Now Starting '+ self.args.experiment_name)
        logger.log('run id is '+ self.args.run_name)
        logger.log(cfg)

        mlflow.set_experiment(self.args.experiment_name)
        mlflow.start_run(run_name=self.args.run_name)
        dct = asdict(cfg)
        for key_0 in dct:
            for key, value in dct[key_0].items():
                mlflow.log_param(key_0+'_'+key, value)

        return cfg, logger
    
    def start_experiment(self,cfg, logger):
        trainer = get_trainer(cfg, logger)
        if self.args.mode == 'train':
            trainer.train()

        elif self.args.mode == 'test':
            trainer.test()       

    def end_experiment(self):


        mlflow.log_artifact(self.log_path)
        mlflow.log_artifact(self.config_path)
        mlflow.end_run()

