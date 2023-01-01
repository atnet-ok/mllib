### for training class
from abc import ABCMeta, abstractmethod

from mllib.src.model import *
from mllib.src.optimizer import *
from torch.utils.data import DataLoader
import torch

class Algorithm(metaclass=ABCMeta):

    @abstractmethod
    def train(self) -> dict:
        pass  

    @abstractmethod
    def test(self) -> dict:
        pass  

class SimpleDeepLerning(Algorithm):
    def __init__(self, cfg) -> None:
        super().__init__()
        model = get_model(cfg)
        optimizer, model = get_optimizer(cfg, model)

        self.class_num = cfg.dataset.class_num
        self.epoch = cfg.algo.epoch
        self.device = cfg.algo.device
        self.optimizer = optimizer
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.amp =False #True
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        

    def update(self, data_loader:DataLoader) -> dict:

        self.model.train()
        loss_total = 0
        size = len(data_loader.dataset)
        num_batches = len(data_loader)

        for data, label in data_loader:
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

    def test(self, data_loader:DataLoader) -> dict:
        self.model.eval()
        metrics = 0
        size = len(data_loader.dataset)
        num_batches = len(data_loader)

        with torch.no_grad():
            for data, label in data_loader:
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

    def train(self, train_loader, test_loader) -> dict:
        for epoch in range(self.epoch):
            print(f"Epoch {epoch+1}\n-------------------------------")
            loss = self.update(train_loader)
            #metrics = self.test(test_loader)
            print(f"loss:{loss}")
            #print(f"metrics:{metrics}")

class Sklern(Algorithm):
    def __init__(self,model) -> None:
        super().__init__()
        self.model = model

    def train(self, data_loader:DataLoader) -> dict:
        pass  


    def test(self, data_loader:DataLoader) -> dict:
        pass  

def get_algo(cfg):
    if cfg.algo.name=="SimpleDeepLerning":
        algo = SimpleDeepLerning(cfg)
    return algo