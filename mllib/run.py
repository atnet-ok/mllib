from mllib.src.config import *
from mllib.src.trainer import *
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-cfg',
        '--cfg_path', 
        default="mllib/config/default.yaml", 
        type=str, 
        nargs='+',
        help='path to config file.'
        )
    config_path=parser.cfg_path
    cfg = get_config(
        config_path=config_path
        )
    algo = get_trainer(cfg)
    algo.train()
    algo.test()
