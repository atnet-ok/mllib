from dataclasses import dataclass, fields
from typing import get_type_hints

import datetime
import yaml
import torch
import numpy as np
import random

@dataclass
class RecursiveDataclass:
    pass
    @classmethod
    def from_dict(cls, src: dict):
        kwargs = dict()
        field_dict= {field.name: field for field in fields(cls)}
        field_type_dict: dict[str, type] = get_type_hints(cls)
        for src_key, src_value in src.items():
            assert src_key in field_dict, "Invalid Data Structure"
            field = field_dict[src_key]
            field_type = field_type_dict[field.name]
            if issubclass(field_type, RecursiveDataclass):
                kwargs[src_key] = field_type.from_dict(src_value)
            else:
                kwargs[src_key] = src_value
        return cls(**kwargs)

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
