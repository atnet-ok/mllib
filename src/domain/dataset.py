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
from typing import Union,Optional,Callable

from src.common.preprocess import *
from src.config import dataset_cfg

class MllibDataset(object):
    def __init__(self, dataset_cfg:dataset_cfg) -> None:
        super().__init__()
        self.cfg = dataset_cfg

    def _load_dataset(self,phase:str):

        transform = None
        target_transform = None

        if self.cfg.name in bildin_dataset_dict.keys():

            dataset = bildin_dataset_dict[self.cfg.name](
                    root=self.cfg.load_dir,
                    transform=transform,
                    target_transform=target_transform,
                    download=True
                    )
            val_size = int(self.cfg.eval_rate * len(dataset))
            train_size = len(dataset) - val_size
            dataset_train, dataset_test = random_split(
                    dataset, 
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(self.cfg.seed)
                )
            
            return dataset_train if phase=="train" else dataset_test

            
        elif self.cfg.name in custom_dataset_dict.keys():
            dataset = custom_dataset_dict[self.cfg.name](
                    root=self.cfg.load_dir,
                    transform=transform,
                    target_transform=target_transform,
                    phase=phase
                    )
            return dataset
        
        else:
            print("!!!")

    def get_dataset(self, phase:str):
        dataset = self._load_dataset(phase)
        return dataset

    def get_dataloader(self, phase:str):

        dataset = self._load_dataset(phase)

        train_batch_size = self.cfg.batch_size_train
        eval_batch_size = self.cfg.batch_size_eval
        num_workers=self.cfg.num_workers

        if phase=='train':
            loader = DataLoader(
                dataset, 
                batch_size=train_batch_size, 
                shuffle=True, 
                num_workers=num_workers, 
                pin_memory=True
            )
        else:
            loader = DataLoader(
                dataset, 
                batch_size=eval_batch_size,
                shuffle=False, 
                num_workers=num_workers, 
                pin_memory=True
            )

        return loader


class CustomDataset(Dataset):
    def __init__(self, 
                 root: str, 
                 phase: bool = True, 
                 transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None, 
                 ) -> None:
        super().__init__()
        self.load_dir = root
        self.phase = phase
        self.transform = transform
        self.target_transform = target_transform




bildin_dataset_dict = {
    "MNIST":MNIST,
    "Country211":Country211,
    "DTD":DTD
}

custom_dataset_dict = {
    
}