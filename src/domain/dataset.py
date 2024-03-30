from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageNet,MNIST,FashionMNIST,Caltech101

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

        self.dataset =dataset_dict[self.cfg.name](
                root=None,
                train= True if phase=="train" else False,
                transform=transform,
                target_transform=target_transform,
                download=None
        )

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


class OriginalDataset(Dataset):
    def __init__(self, 
                 root: str, 
                 train: bool = True, 
                 transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None, 
                 download: bool = False) -> None:
        super().__init__()
        self.dataset_path = root
        self.is_train = train
        self.is_download = download
        self.transform = transform
        self.target_transform = target_transform


dataset_dict = {
    "FashionMNIST":FashionMNIST,
    "MNIST":MNIST,
    "Caltech101":Caltech101,
    "ImageNet":ImageNet
}
