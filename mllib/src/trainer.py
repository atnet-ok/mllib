### for training class
from abc import ABCMeta, abstractmethod

from mllib.src.data import *
from mllib.src.model import *
from mllib.src.optimizer import *
from sklearn.metrics import classification_report
import torch
import numpy as np
import logging
 
logger = logging.getLogger('LoggingTest')

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
        self.scheduler = get_scheduler(cfg,optimizer)

        self.class_num = cfg.data.class_num
        self.epoch = cfg.train.epoch
        self.device = cfg.train.device
        self.amp =cfg.train.amp #True

        self.optimizer = optimizer
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
        metrics_dict = classification_report(
            y_true, 
            y_pred, 
            output_dict=True
            )
        metrics_dict.update(loss_dct)
        return metrics_dict

    def train(self) -> dict:
        for epoch in range(self.epoch):
            print(f"--------------------------------------")
            print(f"Epoch {epoch+1}")
            train_dct = self._update(self.dl_train, train_mode=True)
            self.scheduler.step(epoch+1)
            eval_dct = self._update(self.dl_eval, train_mode=False)
            for key,value in train_dct.items():
                if key in ["accuracy","loss_total"]:
                    print(f"{key}_train:{value}")
            for key,value in eval_dct .items():
                if key in ["accuracy","loss_total"]:
                    print(f"{key}_eval:{value}")

    def test(self) -> dict:
        eval_dict = self._update(self.dl_eval, train_mode=False)
        return eval_dict

class Sklern(Trainer):
    def __init__(self, cfg) -> None:
        super().__init__()

    def _dataset2np(self,dataset):
        data_s = []
        label_s = []
        for i in range(dataset.__len__()):
            data,label = dataset.__getitem__(i)
            data_s.append(data)
            label_s.append(label)
        return np.array(data_s) , np.array(label_s) 

    def train(self) -> dict:
        pass  

    def test(self) -> dict:
        pass  

trainer_dct = {
    "DeepLerning":DeepLerning,
    "Sklern":Sklern
}

def get_trainer(cfg):
    if cfg.train.name in trainer_dct.keys():
        trainer = trainer_dct[cfg.train.name](cfg)
    else:
        raise Exception(f'{cfg.train.name} in not implemented')
    
    return trainer