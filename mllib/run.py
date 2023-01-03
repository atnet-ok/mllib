import sys
sys.path.append(__file__.replace("mllib/run.py",''))

from mllib.src.trainer import *
from mllib.src.utils import *


import argparse

if __name__=='__main__':
    # recieve args from input
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', '--experiment_name',  default="test", type=str)
    parser.add_argument('-id', '--run_id', default=date2str(), type=str)
    parser.add_argument('-m', '--mode', default='train', type=str)
    parser.add_argument('-cfg', '--cfg_dir', default="config/database/", type=str)
    parser.add_argument('-model', '--model_dir', default="model/", type=str)
    parser.add_argument('-log', '--log_dir', default="log/", type=str)
    args = parser.parse_args()

    # load config file.
    cfg, logger = start_experiment(args)

    # wake up trainer
    

    if args.mode == 'train':
        trainer = get_trainer(cfg, logger)
        model = trainer.train()
        metrics = trainer.test()
    elif args.mode == 'test':
        trainer = get_trainer(cfg, logger)
        model = None
        metrics = trainer.test()       

    end_experiment(args, logger, model, metrics)
