from dataclasses import dataclass
from dllib.common.utils import *



@dataclass
class model_cfg:
    name: str="tf_efficientnet_b7"
    pre_train:bool =True
    in_chans:int=1
    out_dim:int=10
    others:dict = None


@dataclass
class dataset_cfg:
    name:str="MNIST"
    # data_dim:tuple = None # eg. for 256*256 color imgae, define as (3,256,256)
    eval_rate:float= 0.2
    root_dir:str="/mnt/d/data/"
    others = {"img_size":224}

@dataclass
class dataloader_cfg:
    batch_size_train:int = 32
    batch_size_eval:int = 32
    num_workers:int=4

@dataclass
class optimizer_cfg:
    name:str="adam"
    lr:float=0.0004
    wd:float=0.0001
    momentum:float=0.9
    scheduler:str="cosine_warmup"
    sche_cycle:int=40
    others:dict = None

@dataclass
class metrics_cfg:
    task:str="classification"

@dataclass
class trainer_cfg:
    seed:int=42
    epoch:int=50
    device:str="cuda:0"
    amp:bool=True
    others:dict = None
    metrics:metrics_cfg=metrics_cfg()
    optimizer:optimizer_cfg=optimizer_cfg()
    dataset:dataset_cfg=dataset_cfg()
    model:model_cfg=model_cfg()
    dataloader:dataloader_cfg=dataloader_cfg()

@dataclass
class logger_cfg:
    log_uri:str="./.log/"
    experiment_name:str="test"
    run_name:str="test"

# def get_config(config_path):

#     dct = yaml2dct(config_path)
#     cfg = config.from_dict(dct)
#     return cfg

# def save_config(cfg:config, config_id:str=date2str(),save_dir='config/'):
#     dct = asdict(cfg)
#     save_path = os.path.join(save_dir, f"{config_id}.yaml")
#     dct2yaml(dct, save_path)
#     return save_path