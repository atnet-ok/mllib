from dataclasses import dataclass, field
from dllib.common.utils import *

@dataclass
class model_cfg:
    name: str = "TimmClassifier"
    backbone: str="tf_efficientnet_b7"
    pre_train:bool =True
    in_chans:int=1
    out_dim:int=10
    custom:dict = field(default_factory = dict)

@dataclass
class dataset_cfg:
    name:str="MNIST"
    eval_rate:float= 0.2
    root_dir:str="/mnt/d/data/"
    fold:int=0
    custom:dict = field(default_factory = lambda:{"class_num":10})

@dataclass
class dataloader_cfg:
    batch_size_train:int = 128
    batch_size_eval:int = 128
    num_workers:int=8

@dataclass
class optimizer_cfg:
    name:str="adam"
    lr:float=4e-4
    wd:float=1e-6
    momentum:float=0.9
    scheduler:str="cosine_warmup"
    warmup_t_rate:float=0.12
    warmup_lr_init_rate:float=0.1
    custom:dict = field(default_factory = dict)

@dataclass
class trainer_cfg:
    name:str="BasicTrainer"
    seed:int=42
    epoch:int=3
    device:str="cuda:0"
    amp:bool=True
    task:str = "classification"
    optimizer:optimizer_cfg=optimizer_cfg()
    dataset:dataset_cfg=dataset_cfg()
    model:model_cfg=model_cfg()
    dataloader:dataloader_cfg=dataloader_cfg()
    custom:dict = field(default_factory = dict)

@dataclass
class predictor_cfg:
    seed:int=42
    device:str="cuda:0"
    model_path:str="log/mlruns/data/model.pth"
    custom:dict = field(default_factory = dict)


@dataclass
class logger_cfg:
    log_uri:str="log/mlruns"
    experiment_name:str="000_test_expr"
    run_name:str="test_run"

