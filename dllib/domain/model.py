from torch import nn
import torch.nn.functional as F
import timm
import torch
import numpy as np

from dllib.config import model_cfg

class TimmClassifier(nn.Module):
    def __init__(self, backbone, out_dim, in_chans,pre_train=True):
        super(TimmClassifier, self).__init__()

        # for timm
        self.backbone =  timm.create_model(
            backbone, 
            pretrained=pre_train, 
            num_classes= 0,
            in_chans=in_chans
        )
        in_features = self.backbone.num_features

        self.head = nn.Linear(in_features, out_dim)

    def forward(self, x):
        embedding = self.backbone(x)
        y = self.head(embedding)
        self.embedding = embedding

        return y


def get_model(model_cfg:model_cfg):

    if model_cfg.name == "TimmClassifier":
        model = TimmClassifier(
            backbone=model_cfg.backbone, 
            pre_train=model_cfg.pre_train,
            in_chans=model_cfg.in_chans,
            out_dim=model_cfg.out_dim
            )
    
    return model