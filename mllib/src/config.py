from dataclasses import dataclass, fields, asdict
from mllib.src.utils import *
import os

@dataclass
class train_cfg(RecursiveDataclass):
    name: str="SimpleDeepLerning"
    epoch:int =30
    optimizer:str="sgd"
    lr:float=1e-4
    wd:float=1e-4
    momentum:float=0.9
    device:str="cuda:0"
    amp:bool=True
    scheduler:str="none"

@dataclass
class model_cfg(RecursiveDataclass):
    name: str="tf_efficientnet_b7"
    pre_train:bool =True
    in_chans:int=1

@dataclass
class data_cfg(RecursiveDataclass):
    name:str="MNIST"
    batch_size_train:int = 32
    batch_size_eval:int = 16
    class_num:int = 10
    data_size:int = 224 
    domain_src:str = None
    domain_trg:str = None

@dataclass
class config(RecursiveDataclass):
    model: model_cfg = model_cfg()
    data: data_cfg = data_cfg()
    train: train_cfg= train_cfg()

def get_config(config_path:str="mllib/config/default.yaml"):
    dct = yaml2dct(config_path)
    cfg = config.from_dict(dct)
    return cfg

def save_config(cfg:config, save_dir:str="mllib/config/database"):
    now = date2str()
    dct = asdict(cfg)
    save_path = os.path.join(
            save_dir,
            f"{now}.yaml"
            )
    dct2yaml(
        dct, 
        save_path
        )

    return save_path