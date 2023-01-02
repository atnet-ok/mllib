import datetime
import yaml
import torch
import numpy as np
import random

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def yaml2dct(yaml_path):
    with open(yaml_path) as file:
        dct = yaml.safe_load(file)
    return dct

def dct2yaml(dct, yaml_path):
    with open(yaml_path, 'w') as file:
        yaml.dump(dct, file)

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
