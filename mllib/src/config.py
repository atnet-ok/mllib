from dataclasses import dataclass, fields, asdict
from mllib.src.utils import *
import os
import yaml

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
    name: str="SimpleDeepLerning"
    seed:int=42
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
    model_trained:str=''

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

def yaml2dct(yaml_path):
    with open(yaml_path) as file:
        dct = yaml.safe_load(file)
    return dct

def dct2yaml(dct, yaml_path):
    with open(yaml_path, 'w') as file:
        yaml.dump(dct, file)

def get_config(config_path:str="config/default.yaml"):
    dct = yaml2dct(config_path)
    cfg = config.from_dict(dct)
    return cfg

def save_config(cfg:config, save_dir:str="config/database"):
    now = date2str()
    dct = asdict(cfg)
    save_path = os.path.join(save_dir, f"{now}.yaml")
    dct2yaml(dct, save_path)

    return save_path