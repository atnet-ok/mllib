from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DTD,Country211,MNIST

from torch.utils.data import random_split
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pandas as pd
import os
from typing import Union,Optional,Callable

from dllib.common.preprocess import get_preprocess
from dllib.config import dataset_cfg,dataloader_cfg


# class MllibDataset(Dataset):
#     def __init__(self, 
#                 root_dir:str,
#                 preprocess:Optional[Callable],
#                 target_transform:Optional[Callable],
#                 phase:str,
#                 eval_rate:float,
#                 others:dict
#                  ) -> None:

#         pass



class Birdclef2024Dataset(Dataset):
    def __init__(
            self, 
            root_dir:str,
            preprocess:Optional[Callable],
            target_transform:Optional[Callable],
            phase:str,
            eval_rate:float,
            others:dict
                ) -> None:
        
        self.df = pd.read_csv(os.path.join(root_dir,"train_metadata.csv"))



custom_dataset_dict = {
    "Birdclef2024":Birdclef2024Dataset
}

def get_dataset(dataset_cfg:dataset_cfg,phase:str):

    preprocess,target_transform = get_preprocess(dataset_cfg.preprocess)

    if dataset_cfg.name in custom_dataset_dict.keys():
        dataset = custom_dataset_dict[dataset_cfg.name](
                root=dataset_cfg.root_dir,
                preprocess=preprocess,
                target_transform=target_transform,
                phase=phase,
                eval_rate=dataset_cfg.eval_rate,
                others=dataset_cfg.others
                )
        return dataset
    
    else:
        raise Exception(f'{dataset_cfg.name} in not implemented')



def get_dataloader(
        dataset:Dataset,
        dataloader_cfg:dataloader_cfg, 
        phase:str):

    train_batch_size = dataloader_cfg.batch_size_train
    eval_batch_size = dataloader_cfg.batch_size_eval
    num_workers=dataloader_cfg.num_workers

    if phase=='train':
        dataloader = DataLoader(
            dataset, 
            batch_size=train_batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True
        )
    else:
        dataloader = DataLoader(
            dataset, 
            batch_size=eval_batch_size,
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=True
        )

    return dataloader
