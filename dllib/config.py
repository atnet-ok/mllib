from dataclasses import dataclass
from dllib.common.utils import *

@dataclass
class model_cfg:
    name: str="resnet18"
    pre_train:bool =True
    in_chans:int=1
    out_dim:int=10
    others = None

@dataclass
class dataset_cfg:
    name:str="MNIST"
    eval_rate:float= 0.2
    root_dir:str="/mnt/d/data/"
    others = {
        "img_size":224,
        "class_num":10
        }

@dataclass
class dataloader_cfg:
    batch_size_train:int = 128
    batch_size_eval:int = 128
    num_workers:int=4

@dataclass
class optimizer_cfg:
    name:str="adam"
    lr:float=0.0004
    wd:float=0.0001
    momentum:float=0.9
    scheduler:str="cosine_warmup"
    sche_cycle:int=40
    others = None

@dataclass
class trainer_cfg:
    seed:int=42
    epoch:int=5
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

