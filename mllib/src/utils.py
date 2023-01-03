from dataclasses import dataclass, fields
from typing import get_type_hints
from mllib.src.config import *
from mllib.src.logger import *

import datetime
import yaml
import torch
import numpy as np
import random
import os


def date2str():
    dt_now = datetime.datetime.now()
    return dt_now.strftime('%Y%m%d_%H%M_%S')

def fix_randomness(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def start_experiment(args):
    config_path=os.path.join(args.cfg_dir, args.run_id+'.yaml')
    logger = start_logging(args.experiment_name)
    cfg = get_config(config_path=config_path)
    return cfg, logger

def end_experiment(args, logger, model, metrics):
    if model:
        save_model(model, args.model_dir, args.run_id)

