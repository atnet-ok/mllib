from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DTD,Country211,MNIST
from torchvision import transforms

import albumentations as A
import torch.nn.functional as F
from torch.utils.data import random_split
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pandas as pd
import os
import torchaudio
from dllib.config import dataset_cfg,dataloader_cfg
from glob import glob

# https://qiita.com/tomp/items/3bf6d040bbc89a171880
# https://qiita.com/yujimats/items/2078f98655d93e66af30

class Birdclef2024Dataset(Dataset):
    def __init__(
            self, 
            root_dir:str,
            phase:str,
            eval_rate:float,
            img_size,
            seed,
            add_secondary_labels=True
                ) -> None:
        
        mel_spec_params = {
            "sample_rate": 32000,
            "n_mels": 128,
            "f_min": 20,
            "f_max": 16000,
            "n_fft": 2048,
            "hop_length": 512,
            "normalized": True,
            "center" : True,
            "pad_mode" : "constant",
            "norm" : "slaney",
            "onesided" : True,
            "mel_scale" : "slaney"
        }

        self.phase = phase

        image_size = img_size
        top_db = 80
        train_period = 5
        val_period = 5
        N_FOLD = 5
        secondary_coef = 1.0

        fold = seed%N_FOLD

        print(f"fold:{fold}/{N_FOLD}")

        self.sample_rate = mel_spec_params["sample_rate"]
        self.train_duration = train_period * mel_spec_params["sample_rate"]
        self.val_duration = val_period * mel_spec_params["sample_rate"]

        transform = self.get_transform(phase=phase,image_size=image_size)
        
        df = self.load_dataset(phase,root_dir,fold,seed,N_FOLD=5)
        self.df = df
        
        sub = pd.read_csv(os.path.join(root_dir,"birdclef-2024/sample_submission.csv"))
        target_columns = sub.columns.tolist()[1:]
        num_classes = len(target_columns)
        bird2id = {b: i for i, b in enumerate(target_columns)}
        id2bird = {i: b for i, b in enumerate(target_columns)}

        self.bird2id = bird2id
        self.id2bird = id2bird
        self.num_classes = num_classes
        self.secondary_coef = secondary_coef
        self.add_secondary_labels = add_secondary_labels
        self.mel_transform = torchaudio.transforms.MelSpectrogram(**mel_spec_params)
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=top_db)
        self.transform = transform
        self.target_columns = target_columns

    def __len__(self):
        return len(self.df)
    
    def load_dataset(self,phase,root_dir,fold,seed,N_FOLD=5):
        if phase=="test":
            filepath_s = glob(os.path.join(root_dir,'birdclef-2024/test_soundscapes/*.ogg'))
            path_s = [os.path.join(root_dir,'birdclef-2024/test_soundscapes/',filepath) for filepath in filepath_s]
            df = pd.DataFrame()
            df["path"] = path_s
            return df
        
        else:
            df = pd.read_csv(os.path.join(root_dir,'birdclef-2024/train_metadata.csv'))
            df["path"] = os.path.join(root_dir,"birdclef-2024/train_audio/") + df["filename"]
            df["rating"] = np.clip(df["rating"] / df["rating"].max(), 0.1, 1.0)

            skf = StratifiedKFold(n_splits=N_FOLD, random_state=0, shuffle=True)
            df['fold'] = -1
            for ifold, (train_idx, val_idx) in enumerate(skf.split(X=df, y=df["primary_label"].values)):
                df.loc[val_idx, 'fold'] = ifold

            if seed >= 0:
                trn_df = df[df['fold'] != fold].reset_index(drop=True)
                val_df = df[df['fold'] == fold].reset_index(drop=True)
            else:
                trn_df = df.reset_index(drop=True)
                val_df = df[df['fold'] == fold].reset_index(drop=True)

            return  trn_df if phase == "train" else val_df


    def get_transform(self,phase,image_size):
        transforms_train = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Resize(image_size, image_size),
            A.CoarseDropout(max_height=int(image_size * 0.375), max_width=int(image_size * 0.375), max_holes=1, p=0.7),
            A.Normalize()
        ])

        transforms_val = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize()
        ])

        return transforms_train if phase == "train" else transforms_val

    def prepare_target(self, primary_label, secondary_labels):
        secondary_labels = eval(secondary_labels)
        target = np.zeros(self.num_classes, dtype=np.float32)
        if primary_label != 'nocall':
            primary_label = self.bird2id[primary_label]
            target[primary_label] = 1.0
            if self.add_secondary_labels:
                for s in secondary_labels:
                    if s != "" and s in self.bird2id.keys():
                        target[self.bird2id[s]] = self.secondary_coef
        target = torch.from_numpy(target).float()
        return target

    def prepare_spec(self, path):
        wav = read_wav(path,self.sample_rate)
        wav = crop_start_wav(wav, self.train_duration)
        mel_spectrogram = normalize_melspec(self.db_transform(self.mel_transform(wav)))
        mel_spectrogram = mel_spectrogram * 255
        mel_spectrogram = mel_spectrogram.expand(3, -1, -1).permute(1, 2, 0).numpy()
        return mel_spectrogram

    def __getitem__(self, idx):
        if self.phase == "test":
            path = self.df["path"].iloc[idx]
            spec = self.prepare_spec(path)
            res = self.transform(image=spec)
            spec = res['image'].astype(np.float32)
            spec = spec.transpose(2, 0, 1)
            return spec
        
        else:
            path = self.df["path"].iloc[idx]
            primary_label = self.df["primary_label"].iloc[idx]
            secondary_labels = self.df["secondary_labels"].iloc[idx]
            rating = self.df["rating"].iloc[idx]

            spec = self.prepare_spec(path)
            target = self.prepare_target(primary_label, secondary_labels)

            if self.transform is not None:
                res = self.transform(image=spec)
                spec = res['image'].astype(np.float32)
            else:
                spec = spec.astype(np.float32)

            spec = spec.transpose(2, 0, 1)
            return {"spec": spec, "target": target, 'rating': rating}

def normalize_melspec(X, eps=1e-6):
    mean = X.mean((1, 2), keepdim=True)
    std = X.std((1, 2), keepdim=True)
    Xstd = (X - mean) / (std + eps)

    norm_min, norm_max = (
        Xstd.min(-1)[0].min(-1)[0],
        Xstd.max(-1)[0].max(-1)[0],
    )
    fix_ind = (norm_max - norm_min) > eps * torch.ones_like(
        (norm_max - norm_min)
    )
    V = torch.zeros_like(Xstd)
    if fix_ind.sum():
        V_fix = Xstd[fix_ind]
        norm_max_fix = norm_max[fix_ind, None, None]
        norm_min_fix = norm_min[fix_ind, None, None]
        V_fix = torch.max(
            torch.min(V_fix, norm_max_fix),
            norm_min_fix,
        )
        V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
        V[fix_ind] = V_fix
    return V

def read_wav(path,sample_rate):
    wav, org_sr = torchaudio.load(path, normalize=True)
    wav = torchaudio.functional.resample(wav, orig_freq=org_sr, new_freq=sample_rate)
    return wav


def crop_start_wav(wav, duration_):
    while wav.size(-1) < duration_:
        wav = torch.cat([wav, wav], dim=1)
    wav = wav[:, :duration_]
    return wav



class MNIST_(Dataset):
    def __init__(
            self, 
            root_dir:str,
            phase:str,
            eval_rate:float,
            others:dict
                ) -> None:

        img_size = others["img_size"]
        self.class_num = others["class_num"]
            
        self.dataset = MNIST(
            root=root_dir,
            train=True if phase=="train" else False,
            transform=self.get_preprocesser(),
            download=True
        )

    def get_preprocesser(self):
        transform  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return  transform 

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        img,label= self.dataset.__getitem__(index)
        label = F.one_hot(torch.tensor(label),num_classes=self.class_num)
        return img, label


custom_dataset_dict = {
    "Birdclef2024":Birdclef2024Dataset,
    "MNIST":MNIST_
}

def get_dataset(dataset_cfg:dataset_cfg,phase:str):

    if dataset_cfg.name in custom_dataset_dict.keys():
        dataset = custom_dataset_dict[dataset_cfg.name](
                root_dir=dataset_cfg.root_dir,
                phase=phase,
                eval_rate=dataset_cfg.eval_rate,
                seed=dataset_cfg.seed,
                img_size=dataset_cfg.img_size,
                )
        return dataset
    
    else:
        raise Exception(f'{dataset_cfg.name} in not implemented')



def get_dataloader(
        dataset:Dataset,
        dataloader_cfg:dataloader_cfg, 
        phase:str):

    train_batch_size = dataloader_cfg.batch_size_train
    eval_batch_size = dataloader_cfg.batch_size_eval
    num_workers=dataloader_cfg.num_workers

    if phase=='train':
        dataloader = DataLoader(
            dataset, 
            batch_size=train_batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True
        )
    else:
        dataloader = DataLoader(
            dataset, 
            batch_size=eval_batch_size,
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=True
        )

    return dataloader
