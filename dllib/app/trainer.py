### for training class
from abc import ABCMeta, abstractmethod
from dllib.domain.dataset import *
from dllib.domain.model import *
from dllib.common.optimizer import *
from dllib.common.logger import *
from dllib.domain.metrics import *
import torch
import numpy as np
import warnings


class Trainer(object):
    def __init__(
            self, 
            trainer_cfg:trainer_cfg=trainer_cfg(), 
            logger:Logger=Logger()
            ) -> None:

        # warnings.simplefilter('ignore', DeprecationWarning)
        # warnings.simplefilter('ignore', UserWarning)

        self.cfg = trainer_cfg
        self.logger = logger

        self.model = get_model(self.cfg.model)
        self.metrics, self.criterion = get_metrics(self.cfg.task)

        self.optimizer,  self.scheduler, self.model = get_optimizer(
            self.cfg.optimizer, 
            self.model
            )

        self.epoch = self.cfg.epoch
        self.device = self.cfg.device
        self.amp = self.cfg.amp #True

        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        #self.model = torch.compile(self.model) #for pytorch 2.0

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        
    def _update(self, dataloader:DataLoader ,phase:str='train', epoch =None) -> dict:

        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        loss_dct = {"loss_total":0}
        y_true = []
        y_pred = []

        num_batches = len(dataloader)

        for data, label in dataloader:
            self.optimizer.zero_grad()
            data=data.to(torch.float32).to(self.device)
            label=label.to(self.device)

            with torch.set_grad_enabled(phase == 'train'):
                with torch.cuda.amp.autocast(
                    enabled=(self.amp and (phase == 'train'))
                    ):
                    y = self.model(data)
                    loss = self.criterion(y,label)

            if phase == 'train':
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step(epoch+1)

            loss_dct['loss_total'] += loss.item()
            y_true.extend(list(label.detach().cpu().numpy()))
            y_pred.extend(list(y.detach().cpu().numpy()))

        loss_dct['loss_total'] = loss_dct['loss_total']/num_batches

        metrics_dict = self.metrics(y_pred, y_true)
        metrics_and_loss_dict = self.logger.log_metrics(
            metrics_dict, phase, self.score, loss_dct, epoch
            )
        return metrics_and_loss_dict

    def train(self) -> dict:

        dataset_train = get_dataset(self.cfg.dataset,phase='train')
        dataset_eval = get_dataset(self.cfg.dataset,phase='eval')
        dataloader_train = get_dataloader(dataset_train,self.cfg.dataloader, 'train')
        dataloader_eval = get_dataloader(dataset_eval,self.cfg.dataloader, 'eval')

        for epoch in range(self.epoch):
            self.logger.log(f"--------------------------------------")
            self.logger.log(f"Epoch {epoch+1}")
            _ = self._update(dataloader_train, 'train', epoch)
            metrics_and_loss_dict = self._update(dataloader_eval, 'eval', epoch)

            if (metrics_and_loss_dict[f"{self.score}/eval"] - self.best_score)*self.score_direction>0:
                self.best_score = metrics_and_loss_dict[f"{self.score}/eval"]
                self.logger.log_model(self.model)
                print("best model ever!")
 

    def test(self) -> dict:
        dataset_test = get_dataset(self.cfg.model, phase='test')
        dl_test = get_dataloader(dataset_test, self.cfg.model, 'test')
        metrics_and_loss_dict = self._update( dl_test, 'test')

# trainer_dct = {
#     "Trainer":Trainer
# }

# def get_trainer(cfg, logger):
#     if cfg.train.name in trainer_dct.keys():
#         trainer = trainer_dct[cfg.train.name](cfg, logger)
#     else:
#         raise Exception(f'{cfg.train.name} in not implemented')
    
#     return trainer