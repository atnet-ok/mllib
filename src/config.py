from dataclasses import dataclass, fields, asdict
from typing import get_type_hints
from src.common.utils import *

import os


@dataclass
class model_cfg:
    name: str="tf_efficientnet_b7"
    pre_train:bool =True
    in_chans:int=3
    model_trained:str=False
    other:dict = None

@dataclass
class dataset_cfg:
    name:str="MNIST"
    batch_size_train:int = 32
    batch_size_eval:int = 32
    class_num:int = None
    data_size:int = None #listの方が画像やセンサデータにも対応できてよい？
    eval_rate:float= 0.2
    num_workers:int=4
    load_dir:str="/mnt/d/data/"
    seed:int=42
    other:dict = None

@dataclass
class preprocess_cfg:
    name:str="general_image_preprocess"

@dataclass
class optimizer_cfg:
    name:str="adam"
    lr:float=0.0004
    wd:float=0.0001
    momentum:float=0.9
    scheduler:str="cosine_warmup"
    other:dict = None

@dataclass
class trainer_cfg:
    name:str="DLTrainer"
    task:str="classification"
    seed:int=42
    epoch:int=50
    optimizer:optimizer_cfg=optimizer_cfg()
    device:str="cuda:0"
    amp:bool=True
    other:dict = None

@dataclass
class logger_cfg:
    log_dir:str="./log/"
    experiment_name:str="test"
    run_id:str="test"



# def get_config(config_path):

#     dct = yaml2dct(config_path)
#     cfg = config.from_dict(dct)
#     return cfg

# def save_config(cfg:config, config_id:str=date2str(),save_dir='config/'):
#     dct = asdict(cfg)
#     save_path = os.path.join(save_dir, f"{config_id}.yaml")
#     dct2yaml(dct, save_path)
#     return save_path