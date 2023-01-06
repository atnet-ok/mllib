from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageNet,MNIST,FashionMNIST,Caltech101
from torchvision import transforms
from torch.utils.data import random_split
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pandas as pd
import os

from mllib.src.preprocess import get_transform

class CWRUsyn2real(Dataset):
    def __init__(self,domain='real',is_train=True,seed=42) -> None:
        super().__init__()
        if domain == 'real':
            X = np.load("data/CWRU_syn2real/preprocessed/XreallDEenv.npy")
            y = np.load("data/CWRU_syn2real/preprocessed/yreallDEenv.npy")
        elif domain == 'syn':
            X = np.load('data/CWRU_syn2real/preprocessed/XsynallDEenv.npy')
            y = np.load('data/CWRU_syn2real/preprocessed/ysynallDEenv.npy')
        
        X_train, X_eval, y_train, y_eval = train_test_split(
                                                    X, 
                                                    y, 
                                                    stratify=y,
                                                    test_size=0.2, 
                                                    random_state=seed
                                                    )
        if is_train:
            self.X = X_train
            self.y = y_train
        else:
            self.X = X_eval
            self.y = y_eval  

    def __getitem__(self, index):
        data = self.X[index] #[np.newaxis,:]
        label = self.y[index]
        return data, label

    def __len__(self):
        return self.y.shape[0]

class OfficeHome(Dataset):
    def __init__(self, img_size, domain="Art", is_train=True, root = "/mnt/d/data/OfficeHomeDataset/",seed=42) -> None:
        super().__init__()
        df = pd.DataFrame()
        df['path'] = glob(root + "/**/**/*.jpg")
        df['domain'] = df["path"].map(lambda x: x.split("/")[-3])
        df['class'] = df["path"].map(lambda x: x.split("/")[-2])

        self.class_num = len(set(df['class'].to_list()))
        df = df[df["domain"]==domain].reset_index(drop=True)

        train_df, test_df = train_test_split(
            df,  
            test_size=0.2, 
            random_state=seed, 
            stratify=df["class"]
        )
        if is_train:
            self.df = train_df
        else:
            self.df = test_df  

        self.domain = domain
        classes = sorted(list(set(df['class'].to_list())))
        self.label_dct={classes[i]:i  for i in range(self.class_num)}
        self.transform = get_transform(img_size,is_train=is_train)

    def __getitem__(self, index):
        img = self.__get_img(index)
        img = self.transform(img) #torch vision

        label =  self.label_dct[self.df['class'].iloc[index]]
        label_onehot = torch.eye(self.class_num)[label]

        return img, label_onehot

    def __len__(self):
        return self.df.shape[0]

    def __get_img(self,index):
        path = self.df['path'].iloc[index]
        img = Image.open(path) #torch vision
        return img


buildin_dataset = {
    "FashionMNIST":FashionMNIST,
    "MNIST":MNIST,
    "Caltech101":Caltech101,
    "ImageNet":ImageNet
}

def get_dataset(cfg, domain = None):
    if cfg.data.name in buildin_dataset.keys():
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
                ]
                )
        root = "/mnt/d/data"
        dataset= buildin_dataset[cfg.data.name](
            root=root,
            transform=transform,
            download = True
            )

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        dataset_train, dataset_eval= random_split(
            dataset, 
            [
                train_size, 
                val_size
                ]
            )

    elif cfg.data.name =="CWRUsyn2real":
        dataset_train = CWRUsyn2real(
            domain= domain if domain else 'real',
            is_train=True,
            seed=cfg.train.seed
        )
        dataset_eval = CWRUsyn2real(
            domain= domain if domain else 'real',
            is_train=False,
            seed=cfg.train.seed
        )

    elif cfg.data.name =="OfficeHome":
        dataset_train = OfficeHome(
            img_size=cfg.data.data_size,
            domain=domain if domain else "Art",
            is_train=True,
            seed=cfg.train.seed
        )
        dataset_eval = OfficeHome(
            img_size=cfg.data.data_size,
            domain=domain if domain else "Art",
            is_train=False,
            seed=cfg.train.seed
        )

    else:
        raise Exception(f'{cfg.data.name} in not implemented')

    return dataset_train, dataset_eval

def get_dataloader(cfg, dataset_train, dataset_eval):

    train_batch_size = cfg.data.batch_size_train
    eval_batch_size = cfg.data.batch_size_eval

    train_loader = DataLoader(
        dataset_train, 
        batch_size=train_batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    eval_loader = DataLoader(
        dataset_eval, 
        batch_size=eval_batch_size,
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    return train_loader, eval_loader 