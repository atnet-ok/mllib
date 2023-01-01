from dataclasses import dataclass, fields
import json
from typing import get_type_hints

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
class algo_cfg(RecursiveDataclass):
    name: str="SimpleDeepLerning"
    epoch:int =30
    optimizer:str="sgd"
    lr:float=1e-4
    wd:float=1e-4
    device:str="cuda:0"
    test:str="hoge"

@dataclass
class model_cfg(RecursiveDataclass):
    name: str="tf_efficientnet_b7"
    pre_train:bool =True
    in_chans:int=1

@dataclass
class dataset_cfg(RecursiveDataclass):
    name:str="MNIST"
    batch_size_train:int = 32
    batch_size_eval:int = 16
    class_num:int = 10
    data_size:int = 224 

@dataclass
class config(RecursiveDataclass):
    model: model_cfg = model_cfg()
    dataset: dataset_cfg = dataset_cfg()
    algo: algo_cfg= algo_cfg()
    # optimizer:optimizer_cfg=optimizer_cfg()

def get_config(config_path:str="mllib/config/default.json"):
    with open(config_path, 'r') as f:
        dct = json.load(f)
        cfg = config.from_dict(dct)
    return cfg
