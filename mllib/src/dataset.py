from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageNet,MNIST,FashionMNIST
from torchvision import transforms
import torch
from PIL import Image

def get_dataset(cfg):

    if cfg.dataset.name=="FashionMNIST":
        dataset_train= FashionMNIST(
            root="data/",
            transform=transforms.ToTensor(),
            train=True
            )
        dataset_eval= FashionMNIST(
            root="data/",
            transform=transforms.ToTensor(),
            train=False
            )

    elif cfg.dataset.name=="MNIST":
        dataset_train= MNIST(
            root="data/",
            transform=transforms.ToTensor(),
            train=True
            )
        dataset_eval= MNIST(
            root="data/",
            transform=transforms.ToTensor(),
            train=False
            )

    return dataset_train, dataset_eval

def get_dataloader(cfg):
    
    dataset_train, dataset_eval = get_dataset(cfg)

    train_batch_size = cfg.dataset.batch_size_train
    eval_batch_size = cfg.dataset.batch_size_eval

    train_loader = DataLoader(
        dataset_train, 
        batch_size=train_batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )

    eval_loader = DataLoader(
        dataset_eval, 
        batch_size=eval_batch_size,
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )

    return train_loader, eval_loader 