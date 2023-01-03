### for training class
from abc import ABCMeta, abstractmethod
from mllib.src.data import *
from mllib.src.model import *
from mllib.src.optimizer import *
from mllib.src.logger import *
import torch
import numpy as np
import warnings

class Trainer(metaclass=ABCMeta):
    def __init__(self, cfg, logger, model_trained) -> None:
        warnings.simplefilter('ignore', DeprecationWarning)
        warnings.simplefilter('ignore', UserWarning)
        if model_trained:
            self.model = load_model(model_trained)
        else:
            self.model = get_model(cfg)
        self.logger = logger

    @abstractmethod
    def train(self) -> object:
        pass  

    @abstractmethod
    def test(self) -> dict:
        pass  

class DeepLerning(Trainer):
    def __init__(self, cfg, logger, model_trained=None) -> None:
        super().__init__(cfg, logger, model_trained)

        self.optimizer, self.model = get_optimizer(cfg, self.model)
        self.dl_train, self.dl_eval = get_dataloader(cfg)
        self.scheduler, self.optimizer = get_scheduler(cfg, self.optimizer)

        self.class_num = cfg.data.class_num
        self.epoch = cfg.train.epoch
        self.device = cfg.train.device
        self.amp = cfg.train.amp #True

        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        
    def _update(self, dataloader:DataLoader ,train_mode:bool=True, epoch =None) -> dict:

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
                self.scheduler.step(epoch+1)

            loss_dct['loss_total'] += loss.item()
            y_true.extend(list(label.detach().argmax(dim=1).cpu().numpy()))
            y_pred.extend(list(y.detach().argmax(dim=1).cpu().numpy()))

        loss_dct['loss_total'] = loss_dct['loss_total']/num_batches
        metrics_dict = self.logger.log_metrics(y_true,y_pred,loss_dct,epoch)


    def train(self) -> dict:
        for epoch in range(self.epoch):
            self.logger.log(f"--------------------------------------")
            self.logger.log(f"Epoch {epoch+1}")
            self.logger.log(f"training...")
            self._update(self.dl_train, True, epoch)
            self.logger.log(f"evaluating...")
            self._update(self.dl_eval, False, epoch)

        return self.model
 

    def test(self) -> dict:
        metrics_dict = self._update( self.dl_eval, False)
        return metrics_dict

class SKLearn(Trainer):
    def __init__(self, cfg, logger, model_trained=None) -> None:
        super().__init__(cfg, logger, model_trained)

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
        self.logger.log(f"training...")
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_train)
        y_true = self.y_train
        metrics_dict = self.logger.log_metrics(y_true,y_pred)

        return self.model

    def test(self) -> dict:
        self.logger.log(f"evaluating...")
        y_pred = self.model.predict(self.X_eval)
        y_true = self.y_eval
        metrics_dict = self.logger.log_metrics(y_true,y_pred)
        return metrics_dict

trainer_dct = {
    "DeepLerning":DeepLerning,
    "SKLearn":SKLearn
}

def get_trainer(cfg,logger):
    if cfg.train.name in trainer_dct.keys():
        trainer = trainer_dct[cfg.train.name](cfg, logger, cfg.model.model_trained)
    else:
        raise Exception(f'{cfg.train.name} in not implemented')
    
    return trainer