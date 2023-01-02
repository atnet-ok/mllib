### for training class
from abc import ABCMeta, abstractmethod

from mllib.src.data import *
from mllib.src.model import *
from mllib.src.optimizer import *
from mllib.src.logger import *
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import torch
import numpy as np
import logging
import warnings
 
logger = logging.getLogger('LoggingTest')

class Trainer(metaclass=ABCMeta):
    def __init__(self) -> None:
        warnings.simplefilter('ignore', DeprecationWarning)

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
        self.scheduler,self.optimizer = get_scheduler(cfg,optimizer)

        self.class_num = cfg.data.class_num
        self.epoch = cfg.train.epoch
        self.device = cfg.train.device
        self.amp =cfg.train.amp #True

        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        
    def _update(self, dataloader:DataLoader ,train_mode:bool=True) -> dict:

        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        loss_dct = {"loss_total":0}
        y_true = []
        y_pred = []

        num_batches = len(dataloader)

        for data, label in dataloader:
            if len(label.shape) == 1:
                label = torch.eye(self.class_num)[label]
            self.optimizer.zero_grad()
            data=data.to(torch.float32).to(self.device)
            label=label.to(self.device)

            with torch.set_grad_enabled(train_mode):
                with torch.cuda.amp.autocast(enabled=(self.amp and train_mode)):
                    y, _ = self.model(data)
                    loss = self.criterion(y,label)

            if train_mode:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            loss_dct['loss_total'] += loss.item()
            y_true.extend(list(label.detach().argmax(dim=1).cpu().numpy()))
            y_pred.extend(list(y.detach().argmax(dim=1).cpu().numpy()))

        loss_dct['loss_total'] = loss_dct['loss_total']/num_batches
        metrics_dict = calc_metrics(y_true,y_pred,loss_dct)


    def train(self) -> dict:
        for epoch in range(self.epoch):
            print(f"--------------------------------------")
            print(f"Epoch {epoch+1}")
            print(f"training...")
            self._update(self.dl_train, train_mode=True)
            self.scheduler.step(epoch+1)
            print(f"evaluating...")
            self._update(self.dl_eval, train_mode=False)
 

    def test(self) -> dict:
        eval_dict = self._update(self.dl_eval, train_mode=False)
        return eval_dict

class SKLearn(Trainer):
    def __init__(self, cfg) -> None:
        super().__init__()
        if cfg.model.name == "RandomForestClassifier":
            self.model = RandomForestClassifier()
        
        dataset_train, dataset_eval = get_dataset(cfg)
        self.X_train, self.y_train = self._dataset2np(dataset_train)
        self.X_eval, self.y_eval = self._dataset2np(dataset_eval)

    def _dataset2np(self,dataset):
        data_s = []
        label_s = []
        for i in range(dataset.__len__()):
            data,label = dataset.__getitem__(i)
            data_s.append(data)
            label_s.append(label)
        return np.array(data_s) , np.array(label_s) 

    def train(self) -> dict:
        print(f"training...")
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_train)
        y_true = self.y_train
        metrics_dict = calc_metrics(y_true,y_pred)

    def test(self) -> dict:
        print(f"evaluating...")
        y_pred = self.model.predict(self.X_eval)
        y_true = self.y_eval
        metrics_dict = calc_metrics(y_true,y_pred)

trainer_dct = {
    "DeepLerning":DeepLerning,
    "SKLearn":SKLearn
}

def get_trainer(cfg):
    if cfg.train.name in trainer_dct.keys():
        trainer = trainer_dct[cfg.train.name](cfg)
    else:
        raise Exception(f'{cfg.train.name} in not implemented')
    
    return trainer