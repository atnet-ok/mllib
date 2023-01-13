### for training class
from abc import ABCMeta, abstractmethod
from mllib.src.data import *
from mllib.src.model import *
from mllib.src.optimizer import *
from mllib.src.logger import *
from adapt.instance_based import TrAdaBoost
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

    def _dataset2np(self,dataset):
        data_s = []
        label_s = []
        for i in range(dataset.__len__()):
            data,label = dataset.__getitem__(i)
            data_s.append(data)
            label_s.append(label)
        return np.array(data_s) , np.array(label_s) 


class DLTrainer(Trainer):
    def __init__(self, cfg, logger, model_trained=None) -> None:
        super().__init__(cfg, logger, model_trained)

        self.cfg = cfg
        self.optimizer, self.model = get_optimizer(cfg, self.model)
        self.scheduler, self.optimizer = get_scheduler(cfg, self.optimizer)

        self.class_num = cfg.data.class_num
        self.epoch = cfg.train.epoch
        self.device = cfg.train.device
        self.amp = cfg.train.amp #True

        self.model = self.model.to(self.device)
        #self.model = torch.compile(self.model) #for pytorch 2.0
        self.criterion = nn.CrossEntropyLoss()
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
            if len(label.shape) == 1:
                label = torch.eye(self.class_num)[label]
            self.optimizer.zero_grad()
            data=data.to(torch.float32).to(self.device)
            label=label.to(self.device)

            with torch.set_grad_enabled(phase == 'train'):
                with torch.cuda.amp.autocast(enabled=(self.amp and (phase == 'train'))):
                    y, _ = self.model(data)
                    loss = self.criterion(y,label)

            if phase == 'train':
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step(epoch+1)

            loss_dct['loss_total'] += loss.item()
            y_true.extend(list(label.detach().argmax(dim=1).cpu().numpy()))
            y_pred.extend(list(y.detach().argmax(dim=1).cpu().numpy()))

        loss_dct['loss_total'] = loss_dct['loss_total']/num_batches
        metrics_dict = self.logger.log_metrics(y_true,y_pred,phase,loss_dct,epoch)


    def train(self) -> dict:

        dataset_train = get_dataset(self.cfg,phase='train')
        dataset_eval = get_dataset(self.cfg,phase='eval')
        dl_train = get_dataloader(self.cfg, 'train',dataset_train)
        dl_eval = get_dataloader(self.cfg, 'eval',dataset_eval)

        for epoch in range(self.epoch):
            self.logger.log(f"--------------------------------------")
            self.logger.log(f"Epoch {epoch+1}")
            self._update(dl_train, 'train', epoch)
            self._update(dl_eval, 'eval', epoch)

        return self.model
 

    def test(self) -> dict:
        dataset_test = get_dataset(self.cfg,phase='eval')
        dl_test = get_dataloader(self.cfg, 'eval', dataset_test)
        metrics_dict = self._update( dl_test, 'test')
        return metrics_dict

class MLTrainer(Trainer):
    def __init__(self, cfg, logger, model_trained=None) -> None:
        super().__init__(cfg, logger, model_trained)
        self.cfg = cfg

    def train(self) -> dict:
        dataset_train = get_dataset(self.cfg,phase='train', eval_rate=self.cfg.data.eval_rate)
        dataset_eval = get_dataset(self.cfg,phase='eval', eval_rate=self.cfg.data.eval_rate)

        X_train, y_train = self._dataset2np(dataset_train)
        X_eval, y_eval = self._dataset2np(dataset_eval)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_train)
        y_true = y_train
        _ = self.logger.log_metrics(y_true,y_pred,'train')

        y_pred = self.model.predict(X_eval)
        y_true = y_eval
        _ = self.logger.log_metrics(y_true,y_pred,'eval')

        return self.model

    def test(self) -> dict:
        phase='test'
        dataset_test = get_dataset(self.cfg,phase=phase, eval_rate=self.cfg.data.eval_rate)
        X_test, y_test = self._dataset2np(dataset_test)

        y_pred = self.model.predict(X_test)
        y_true = y_test
        metrics_dict = self.logger.log_metrics(y_true,y_pred,phase)
        return metrics_dict

class MLSDATrainer(Trainer):
    def __init__(self, cfg, logger, model_trained=None) -> None:
        super().__init__(cfg, logger, model_trained)
        self.cfg = cfg

        dataset_train_trg = get_dataset(self.cfg, phase='train', domain=self.cfg.data.domain_trg,eval_rate=self.cfg.data.eval_rate)
        self.X_train_trg, self.y_train_trg = self._dataset2np(dataset_train_trg)

        if cfg.train.da_method == 'TrAdaBoost':
            self.model = TrAdaBoost(
                self.model, 
                n_estimators=10, 
                Xt=self.X_train_trg, 
                yt=self.y_train_trg, 
                random_state=cfg.train.seed
            )
        else:
            raise Exception(f'{cfg.train.da_method} in not implemented')

    def train(self) -> dict:


        dataset_train_src = get_dataset(self.cfg, phase='train', domain=self.cfg.data.domain_src, eval_rate=self.cfg.data.eval_rate)
        #dataset_eval_src = get_dataset(self.cfg, phase='eval', domain=self.cfg.data.domain_src,eval_rate=self.cfg.data.eval_rate)
        dataset_eval_trg = get_dataset(self.cfg, phase='eval', domain=self.cfg.data.domain_trg,eval_rate=self.cfg.data.eval_rate)

        X_train_src, y_train_src = self._dataset2np(dataset_train_src)
        #X_eval_src, y_eval_src = self._dataset2np(dataset_eval_src)
        X_eval_trg, y_eval_trg = self._dataset2np(dataset_eval_trg)

        self.model.fit(X_train_src, y_train_src)

        y_pred = self.model.predict(self.X_train_trg)
        y_true = self.y_train_trg
        _ = self.logger.log_metrics(y_true,y_pred,'train')

        y_pred = self.model.predict(X_eval_trg)
        y_true = y_eval_trg
        _ = self.logger.log_metrics(y_true,y_pred,'eval')


        return self.model

    def test(self) -> dict:
        phase='test'
        dataset_test_trg = get_dataset(self.cfg, phase, domain=self.cfg.data.domain_trg,eval_rate=self.cfg.data.eval_rate)
        X_test_trg, y_test_trg = self._dataset2np(dataset_test_trg)

        y_pred = self.model.predict(X_test_trg)
        y_true = y_test_trg
        metrics_dict = self.logger.log_metrics(y_true,y_pred,phase)
        return metrics_dict


trainer_dct = {
    "DLTrainer":DLTrainer,
    "MLTrainer":MLTrainer,
    "MLSDATrainer":MLSDATrainer
}

def get_trainer(cfg,logger):
    if cfg.train.name in trainer_dct.keys():
        trainer = trainer_dct[cfg.train.name](cfg, logger, cfg.model.model_trained)
    else:
        raise Exception(f'{cfg.train.name} in not implemented')
    
    return trainer