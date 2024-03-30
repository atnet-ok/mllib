from dataclasses import dataclass, fields, asdict
from typing import get_type_hints
from src.common.utils import *

import os


@dataclass
class model_cfg:
    name: str=None
    pre_train:bool =None
    in_chans:int=None
    model_trained:str=None
    other:dict = None

@dataclass
class dataset_cfg:
    name:str=None
    batch_size_train:int = None
    batch_size_eval:int = None
    class_num:int = None
    data_size:int = None #listの方が画像やセンサデータにも対応できてよい？
    eval_rate:float= None
    num_workers:int=None
    other:dict = None
    seed:int=None

@dataclass
class optimizer_cfg:
    name:str=None
    lr:float=None
    wd:float=None
    momentum:float=None
    scheduler:str=None

@dataclass
class train_cfg:
    name:str=None
    task:str=None
    seed:int=None
    epoch:int =None
    optimizer:optimizer_cfg=None
    device:str=None
    amp:bool=None
    method:str=None 
    other:dict = None

@dataclass
class logger_cfg:
    log_dir:str=None
    experiment_name:str=None
    run_id:str=None



# def get_config(config_path):

#     dct = yaml2dct(config_path)
#     cfg = config.from_dict(dct)
#     return cfg

# def save_config(cfg:config, config_id:str=date2str(),save_dir='config/'):
#     dct = asdict(cfg)
#     save_path = os.path.join(save_dir, f"{config_id}.yaml")
#     dct2yaml(dct, save_path)
#     return save_path