import sys
sys.path.append(__file__.replace("mllib/run.py",''))

from mllib.src.manager import *
from mllib.src.utils import *

import argparse

if __name__=='__main__':
    # recieve args from input
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp', '--experiment_name',  default="000_test", type=str)
    parser.add_argument('-run', '--run_name', default='000_default', type=str)
    parser.add_argument('-m', '--mode', default='train', type=str)
    parser.add_argument('-cfg', '--cfg_dir', default="config/", type=str)
    parser.add_argument('-model', '--model_dir', default="model/", type=str)
    parser.add_argument('-log', '--log_dir', default="log/", type=str)
    args = parser.parse_args()


    manager = Manager(args)
    cfg, logger = manager.set_experiment()
    manager.start_experiment(cfg, logger)
    manager.end_experiment()
