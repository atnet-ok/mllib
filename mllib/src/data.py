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

class MllibDataset(Dataset):
    def __init__(self, phase, eval_rate, seed) -> None:
        super().__init__()
        self.phase = phase
        self.eval_rate = eval_rate
        self.seed = seed

class CWRUsyn2real(MllibDataset):
    def __init__(self, domain='real', eval_rate=0.2, phase='train',seed=42) -> None:
        super().__init__( phase, eval_rate, seed)
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
                                                    test_size=eval_rate, 
                                                    random_state=seed
                                                    )
        if phase=='train':
            self.X = X_train
            self.y = y_train
        else:
            self.X = X_eval
            self.y = y_eval  

    def __getitem__(self, index):
        data = self.X[index] #[np.newaxis,:]
        label = self.y[index]
        label = torch.eye(self.class_num)[label]
        return data, label

    def __len__(self):
        return self.y.shape[0]

class OfficeHome(MllibDataset):
    def __init__(self, img_size, domain="Art", eval_rate=0.2, phase='train', root = "/mnt/d/data/OfficeHomeDataset/",seed=42) -> None:
        super().__init__( phase, eval_rate, seed)
        df = pd.DataFrame()
        df['path'] = glob(root + "/**/**/*.jpg")
        df['domain'] = df["path"].map(lambda x: x.split("/")[-3])
        df['class'] = df["path"].map(lambda x: x.split("/")[-2])

        self.class_num = len(set(df['class'].to_list()))
        df = df[df["domain"]==domain].reset_index(drop=True)

        train_df, test_df = train_test_split(
            df,  
            test_size=eval_rate, 
            random_state=seed, 
            stratify=df["class"]
        )
        if phase=='train':
            self.df = train_df
        else:
            self.df = test_df  

        self.domain = domain
        classes = sorted(list(set(df['class'].to_list())))
        self.label_dct={classes[i]:i  for i in range(self.class_num)}
        self.transform = get_transform(img_size,phase=phase)

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

def get_dataset(cfg, phase):
    # if cfg.data.name in buildin_dataset.keys():
    #     transform = transforms.Compose(
    #         [
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5, ), (0.5, ))
    #             ]
    #             )
    #     root = __file__.replace("mllib/src/data.py",'')+'data'
    #     dataset= buildin_dataset[cfg.data.name](
    #         root=root,
    #         transform=transform,
    #         download = True
    #         )

    #     train_size = int((1-eval_rate) * len(dataset))
    #     val_size = len(dataset) - train_size
    #     dataset_train, dataset_eval= random_split(
    #         dataset, 
    #         [
    #             train_size, 
    #             val_size
    #             ]
    #         )

    #     if phase=='train':
    #         dataset = dataset_train
    #     else:
    #         dataset = dataset_eval

    if cfg.data.name =="CWRUsyn2real":
        dataset = CWRUsyn2real(
            phase=phase,
            seed=cfg.train.seed,
            eval_rate=cfg.data.eval_rate,
        )


    elif cfg.data.name =="OfficeHome":
        dataset = OfficeHome(
            phase=phase,
            seed=cfg.train.seed,
            eval_rate=cfg.data.eval_rate,
            img_size=cfg.data.data_size
        )

    else:
        raise Exception(f'{cfg.data.name} in not implemented')

    return dataset

def get_dataloader(cfg, phase, dataset):

    train_batch_size = cfg.data.batch_size_train
    eval_batch_size = cfg.data.batch_size_eval

    if phase=='train':
        loader = DataLoader(
            dataset, 
            batch_size=train_batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True
        )
    else:
        loader = DataLoader(
            dataset, 
            batch_size=eval_batch_size,
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )

    return loader