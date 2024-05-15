from dataclasses import dataclass
from dllib.common.utils import *

@dataclass
class model_cfg:
    name: str = ""
    backbone: str="eca_nfnet_l0"
    pre_train:bool =True
    in_chans:int=3
    out_dim:int=182
    others = None

@dataclass
class dataset_cfg:
    name:str="Birdclef2024"
    eval_rate:float= 0.2
    root_dir:str="/mnt/d/data/"
    others = {
        "img_size":256,
        "seed":3
        }

@dataclass
class dataloader_cfg:
    batch_size_train:int = 128
    batch_size_eval:int = 128
    num_workers:int=4

@dataclass
class optimizer_cfg:
    name:str="adam"
    lr:float=4e-4
    wd:float=1e-6
    momentum:float=0.9
    scheduler:str="cosine_warmup"
    sche_cycle:int=100
    others = None

@dataclass
class trainer_cfg:
    seed:int=42
    epoch:int=100
    device:str="cuda:0"
    amp:bool=True
    task:str = "classification"
    optimizer:optimizer_cfg=optimizer_cfg()
    dataset:dataset_cfg=dataset_cfg()
    model:model_cfg=model_cfg()
    dataloader:dataloader_cfg=dataloader_cfg()
    others = None

@dataclass
class logger_cfg:
    log_uri:str="/mnt/d/log/"
    experiment_name:str="test_expr"
    run_name:str="test_run"

