### for training class
from dllib.domain.dataset import get_dataset,get_dataloader
from dllib.domain.model import get_model
from dllib.common.optimizer import get_optimizer
from dllib.domain.metrics import get_metrics
from dllib.common.logger import Logger
from dllib.config import *

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np

class Trainer(object):
    def __init__(
            self, 
            trainer_cfg:trainer_cfg=trainer_cfg(), 
            logger:Logger=Logger()
            ) -> None:

        self.cfg = trainer_cfg
        self.logger = logger
        self.task = self.cfg.task

        self.model = get_model(self.cfg.model, self.task)
        self.get_metrics, self.criterion = get_metrics(self.task)
        self.best_loss = 1000

        self.optimizer, self.scheduler, self.model = get_optimizer(
            self.cfg.optimizer, 
            self.model
            )

        self.epoch = self.cfg.epoch
        self.device = self.cfg.device
        self.amp = self.cfg.amp 

        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        #self.model = torch.compile(self.model) #for pytorch 2.0

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        
    def _update(self, dataloader:DataLoader ,phase:str='train', epoch:int =None) -> dict:

        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        loss = 0
        y_true = []
        y_pred = []

        num_batches = len(dataloader)

        for data, label in tqdm(dataloader):
            self.optimizer.zero_grad()
            data=data.to(torch.float32).to(self.device)
            label=label.to(torch.float32).to(self.device)

            with torch.set_grad_enabled(phase == 'train'):
                with torch.cuda.amp.autocast(
                    enabled=(self.amp and (phase == 'train'))
                    ):
                    y = self.model(data)
                    loss_t = self.criterion(y,label)

            if phase == 'train':
                self.scaler.scale(loss_t).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step(epoch+1)

            loss += loss_t.item()
            y_true.extend(list(label.detach().cpu().numpy()))
            y_pred.extend(list(y.detach().cpu().numpy()))

        loss = loss/num_batches
        metrics = self.get_metrics(y_pred, y_true)

        return metrics, loss

    def train(self):

        dataset_train = get_dataset(self.cfg.dataset,phase='train')
        dataset_eval = get_dataset(self.cfg.dataset,phase='eval')
        dataloader_train = get_dataloader(dataset_train,self.cfg.dataloader, 'train')
        dataloader_eval = get_dataloader(dataset_eval,self.cfg.dataloader, 'eval')

        for epoch in range(self.epoch):
            print(f"--------------------------------------")
            print(f"Epoch {epoch+1}")

            for phase in ["train", "eval"]:
                metrics, loss = self._update(dataloader_train, phase, epoch)
                print(f"{phase}/loss:{loss}")
                print(f"{phase}/metrics:{metrics}")


            if self.best_loss > loss:
                self.best_loss = loss
                self.logger.log_model(model=self.model,model_name="{:03}_epoch".format(epoch))
                print("best model ever!")

    def test(self):
        dataset_test = get_dataset(self.cfg.model, phase='test')
        dl_test = get_dataloader(dataset_test, self.cfg.model, 'test')
        metrics, loss = self._update( dl_test, 'test')

# trainer_dct = {
#     "Trainer":Trainer
# }

# def get_trainer(cfg, logger):
#     if cfg.train.name in trainer_dct.keys():
#         trainer = trainer_dct[cfg.train.name](cfg, logger)
#     else:
#         raise Exception(f'{cfg.train.name} in not implemented')
    
#     return trainer