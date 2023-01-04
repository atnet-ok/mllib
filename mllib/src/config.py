from dataclasses import dataclass, fields, asdict
from typing import get_type_hints
from mllib.src.utils import *

import os


# https://zenn.dev/yosemat/articles/2fce02d2ad0794
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

@dataclass
class train_cfg(RecursiveDataclass):
    name:str=None
    seed:int=None
    epoch:int =None
    optimizer:str=None
    lr:float=None
    wd:float=None
    momentum:float=None
    device:str=None
    amp:bool=None
    scheduler:str=None
    transfer:bool=None
@dataclass
class model_cfg(RecursiveDataclass):
    name: str=None
    pre_train:bool =None
    in_chans:int=None
    model_trained:str=None

@dataclass
class data_cfg(RecursiveDataclass):
    name:str=None
    batch_size_train:int = None
    batch_size_eval:int = None
    class_num:int = None
    data_size:int = None
    src:str = None
    trg:str = None

@dataclass
class config(RecursiveDataclass):
    model: model_cfg = model_cfg()
    data: data_cfg = data_cfg()
    train: train_cfg= train_cfg()

def get_config(config_path:str="config/000_default.yaml"):
    dct = yaml2dct(config_path)
    cfg = config.from_dict(dct)
    return cfg

def save_config(cfg:config, config_id:str=date2str(),save_dir='config/'):
    dct = asdict(cfg)
    save_path = os.path.join(save_dir, f"{config_id}.yaml")
    dct2yaml(dct, save_path)
    return save_path