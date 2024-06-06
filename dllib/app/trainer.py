### for training class
from dllib.config import trainer_cfg
from dllib.domain.dataset import get_dataset, get_dataloader
from dllib.domain.model import get_model
from dllib.common.optimizer import get_optimizer
from dllib.domain.metrics import get_metrics
from dllib.common.logger import Logger
from dllib.config import *

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


class BasicTrainer(object):
    def __init__(
        self, trainer_cfg: trainer_cfg = trainer_cfg(), logger: Logger = Logger()
    ) -> None:
        self.cfg = trainer_cfg
        self.logger = logger
        self.task = self.cfg.task

        self.model = get_model(self.cfg.model)
        self.criterion, self.get_score, self.is_best_model = get_metrics(self.task)
        self.best_score = None

        self.optimizer, self.scheduler, self.model = get_optimizer(
            self.cfg.optimizer, self.model, self.cfg.epoch
        )

        self.epoch = self.cfg.epoch
        self.device = self.cfg.device
        self.amp = self.cfg.amp

        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

    def _load_data(self):
        self.dataset_train = get_dataset(self.cfg.dataset, phase="train")
        self.dataset_eval = get_dataset(self.cfg.dataset, phase="eval")
        self.dataloader_train = get_dataloader(
            self.dataset_train, self.cfg.dataloader, "train"
        )
        self.dataloader_eval = get_dataloader(
            self.dataset_eval, self.cfg.dataloader, "eval"
        )

    def _step(
        self, dataloader: DataLoader, phase: str = "train", epoch: int = None
    ) -> dict:
        if phase == "train":
            self.model.train()
        else:
            self.model.eval()

        loss = 0
        y_true = []
        y_pred = []

        num_batches = len(dataloader)

        for data, label in tqdm(dataloader):
            self.optimizer.zero_grad()
            data = data.to(torch.float32).to(self.device)
            label = label.to(torch.float32).to(self.device)

            with torch.set_grad_enabled(phase == "train"):
                with torch.cuda.amp.autocast(enabled=(self.amp and (phase == "train"))):
                    y = self.model(data)
                    loss_t = self.criterion(y, label)

            if phase == "train":
                self.scaler.scale(loss_t).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step(epoch + 1)

            loss += loss_t.item()
            y_true.extend(list(label.detach().cpu().numpy()))
            y_pred.extend(list(y.detach().cpu().numpy()))

        loss = loss / num_batches
        score = self.get_score(y_pred, y_true)

        return score, loss

    def train(self):
        self._load_data()
        dl_dct = {"train": self.dataloader_train, "eval": self.dataloader_eval}

        for epoch in range(self.epoch):
            print(f"--------------------------------------")
            print(f"Epoch {epoch+1}")

            for phase in ["train", "eval"]:
                score, loss = self._step(dl_dct[phase], phase, epoch)
                print(f"loss/{phase}:{loss}")
                print(f"score/{phase}:{score}")
                self.logger.log_metrics({f"loss/{phase}": loss}, step=epoch)
                self.logger.log_metrics({f"metrics/{phase}": score}, step=epoch)

            if self.is_best_model(self.best_score, score):
                print("best model ever!")
                self.logger.log_model(model=self.model, model_name="model_best")

        self.logger.log_model(model=self.model, model_name="model_last")


def get_trainer(trainer_cfg: trainer_cfg = trainer_cfg(), logger: Logger = Logger()):
    if trainer_cfg.name == "BasicTrainer":
        trainer = BasicTrainer(trainer_cfg, logger)

    else:
        raise Exception(f"{trainer_cfg.name} in not implemented")

    return trainer
