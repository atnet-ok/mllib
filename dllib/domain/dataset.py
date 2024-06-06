from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DTD, Country211, MNIST
from torchvision import transforms

import albumentations as A
import torch.nn.functional as F
from torch.utils.data import random_split
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pandas as pd
import os
import torchaudio
from dllib.config import dataset_cfg, dataloader_cfg
from dllib.domain.preprocess import preprocess_gray_image

# https://qiita.com/tomp/items/3bf6d040bbc89a171880
# https://qiita.com/yujimats/items/2078f98655d93e66af30


class MNIST_(Dataset):
    def __init__(
        self, root_dir: str, phase: str, eval_rate: float, fold: int, custom: dict
    ) -> None:
        self.class_num = custom["class_num"]

        self.dataset = MNIST(
            root=root_dir,
            train=True if phase == "train" else False,
            transform=self.get_preprocesser(),
            download=True,
        )

    def get_preprocesser(self):
        transform = preprocess_gray_image
        return transform

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        img, label = self.dataset.__getitem__(index)
        label = F.one_hot(torch.tensor(label), num_classes=self.class_num)
        return img, label


custom_dataset_dict = {"MNIST": MNIST_}


def get_dataset(dataset_cfg: dataset_cfg, phase: str):
    if dataset_cfg.name in custom_dataset_dict.keys():
        dataset = custom_dataset_dict[dataset_cfg.name](
            root_dir=dataset_cfg.root_dir,
            phase=phase,
            eval_rate=dataset_cfg.eval_rate,
            fold=dataset_cfg.fold,
            custom=dataset_cfg.custom,
        )
        return dataset

    else:
        raise Exception(f"{dataset_cfg.name} in not implemented")


def get_dataloader(dataset: Dataset, dataloader_cfg: dataloader_cfg, phase: str):
    train_batch_size = dataloader_cfg.batch_size_train
    eval_batch_size = dataloader_cfg.batch_size_eval
    num_workers = dataloader_cfg.num_workers

    if phase == "train":
        dataloader = DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return dataloader
