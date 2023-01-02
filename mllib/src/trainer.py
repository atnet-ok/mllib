### for training class
from abc import ABCMeta, abstractmethod

from mllib.src.dataset import *
from mllib.src.model import *
from mllib.src.optimizer import *
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np



class Trainer(metaclass=ABCMeta):

    @abstractmethod
    def train(self) -> dict:
        pass  

    @abstractmethod
    def test(self) -> dict:
        pass  

class DeepLerning(Trainer):
    def __init__(self, cfg) -> None:
        super().__init__()
        model = get_model(cfg)
        optimizer, model = get_optimizer(cfg, model)
        self.dl_train, self.dl_eval = get_dataloader(cfg)

        self.class_num = cfg.dataset.class_num
        self.epoch = cfg.train.epoch
        self.device = cfg.train.device
        self.optimizer = optimizer
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.amp =False #True
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        

    def update(self) -> dict:


        self.model.train()
        loss_total = 0
        size = len(self.dl_train.dataset)
        num_batches = len(self.dl_train)

        for data, label in self.dl_train:
            if len(label.shape) == 1:
                label = torch.eye(self.class_num)[label].to(self.device)
            self.optimizer.zero_grad()
            data=data.to(self.device)
            label=label.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.amp):
                y, _ = self.model(data)
                loss = self.criterion(label,y)

            # self.scaler.scale(loss).backward()
            # self.scaler.step(self.optimizer)
            # self.scaler.update()
            loss.backward()
            self.optimizer.zero_grad()
            self.optimizer.step()
            loss_total += loss.item()

        loss_total = loss_total/num_batches
        
        return loss_total

    def test(self) -> dict:
        self.model.eval()
        metrics = 0
        size = len(self.dl_eval.dataset)
        num_batches = len(self.dl_eval)

        with torch.no_grad():
            for data, label in self.dl_eval:
                if len(label.shape) == 1:
                    label = torch.eye(self.class_num)[label].to(self.device)
                self.optimizer.zero_grad()
                data=data.to(self.device)
                label=label.to(self.device)
                y, _ = self.model(data)
                loss = self.criterion(label,y)
            loss_total += loss.item()

        loss_total = loss_total/num_batches
        return metrics

    def train(self) -> dict:
        for epoch in range(self.epoch):
            print(f"--------------------------------------")
            print(f"Epoch {epoch+1}")
            loss = self.update()
            #metrics = self.test()
            print(f"loss:{loss}")
            #print(f"metrics:{metrics}")

class Sklern(Trainer):
    def __init__(self,model) -> None:
        super().__init__()
        self.model = model

    def train(self) -> dict:
        pass  


    def test(self) -> dict:
        pass  

    def dataset2np(self,dataset):
        data_s = []
        label_s = []
        for i in range(dataset.__len__()):
            data,label = dataset.__getitem__(i)
            data_s.append(data)
            label_s.append(label)
        return np.array(data_s) , np.array(label_s) 

trainer_dct = {
    "DeepLerning":DeepLerning,
    "Sklern":Sklern
}

def get_trainer(cfg):
    if cfg.train.name in trainer_dct.keys():
        algo = trainer_dct[cfg.train.name](cfg)
    else:
        raise Exception(f'{cfg.train.name} in not implemented')
    
    return algo