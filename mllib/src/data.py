from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageNet,MNIST,FashionMNIST,Caltech101
from torchvision import transforms
from torch.utils.data import random_split
import numpy as np

class CWRUsyn2real(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.X = np.load("data/CWRU_syn2real/preprocessed/XreallDEenv.npy")
        self.y = np.load("data/CWRU_syn2real/preprocessed/yreallDEenv.npy")
        pass

    def __getitem__(self, index):
        data = self.X[index][np.newaxis,:]
        label = self.y[index]
        return data, label

    def __len__(self):
        return self.y.shape[0]

buildin_dataset = {
    "FashionMNIST":FashionMNIST,
    "MNIST":MNIST,
    "Caltech101":Caltech101,
    "ImageNet":ImageNet
}

def get_dataset(cfg):


    if cfg.data.name in buildin_dataset.keys():
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
                ]
                )
        root = "/mnt/d/data"
        dataset= buildin_dataset[cfg.data.name](
            root=root,
            transform=transform,
            download = True
            )

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        dataset_train, dataset_eval= random_split(
            dataset, 
            [
                train_size, 
                val_size
                ]
            )

    elif cfg.data.name =="CWRUsyn2real":
        dataset = CWRUsyn2real()
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        dataset_train, dataset_eval= random_split(
            dataset, 
            [
                train_size, 
                val_size
                ]
            )

    else:
        raise Exception(f'{cfg.data.name} in not implemented')

    return dataset_train, dataset_eval

def get_dataloader(cfg):
    
    dataset_train, dataset_eval = get_dataset(cfg)

    train_batch_size = cfg.data.batch_size_train
    eval_batch_size = cfg.data.batch_size_eval

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