import datetime
import torch
import numpy as np
import random
import yaml


def yaml2dct(yaml_path):
    with open(yaml_path) as file:
        dct = yaml.safe_load(file)
    return dct


def dct2yaml(dct, yaml_path):
    with open(yaml_path, "w") as file:
        yaml.dump(dct, file)


def date2str():
    dt_now = datetime.datetime.now()
    return dt_now.strftime("%Y-%m-%d_%H-%M-%S")


def fix_randomness(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
