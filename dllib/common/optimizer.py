import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR
from timm.scheduler import CosineLRScheduler

from dllib.config import optimizer_cfg

def get_optimizer(optimizer_cfg:optimizer_cfg,model,epoch_n=40):
    opt = optimizer_cfg.name
    lr = optimizer_cfg.lr
    weight_decay = optimizer_cfg.wd if optimizer_cfg.wd else 0
    momentum = optimizer_cfg.momentum if optimizer_cfg.momentum else 0

    if opt == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=lr , 
            momentum=momentum, #0.9, 
            weight_decay=weight_decay #1e-4
            )
    elif opt == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=lr ,
            betas=(0.9, 0.999), 
            eps=1e-08, 
            weight_decay=weight_decay #1e-4
            )
    elif opt == 'adamW':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr ,
            betas=(0.9, 0.999), 
            eps=1e-08, 
            weight_decay=weight_decay #1e-4
            )
    elif opt == 'radam':
        optimizer = optim.RAdam(
            model.parameters(), 
            lr=lr , 
            betas=(0.9, 0.999), 
            eps=1e-08, 
            weight_decay=weight_decay,
            )
    elif opt == 'adadelta':
        optimizer = optim.Adadelta(
            model.parameters(), 
            lr=lr, 
            rho=0.9, 
            eps=1e-06, 
            weight_decay=weight_decay)
    elif opt == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(), 
            lr=lr, 
            alpha=0.99, 
            eps=1e-08, 
            weight_decay=weight_decay, 
            momentum=momentum, 
            centered=False)
        
    else:
        raise Exception(f'{opt} in not implemented')    

    sche = optimizer_cfg.scheduler
    sche_cycle = epoch_n #optimizer_cfg.sche_cycle
    warmup_t_rate=optimizer_cfg.warmup_t_rate
    warmup_lr_init_rate=optimizer_cfg.warmup_lr_init_rate

    if sche=='cosine_warmup':
        scheduler = CosineLRScheduler(
                optimizer, 
                t_initial=sche_cycle , 
                lr_min=lr*warmup_lr_init_rate, 
                warmup_t=round(sche_cycle*warmup_t_rate), 
                warmup_lr_init=lr*warmup_lr_init_rate, 
                warmup_prefix=True
                )
        
    elif sche=='cosine':
        scheduler = CosineAnnealingLR(
            optimizer, 
            eta_min=1e-4, 
            T_max= sche_cycle)
        
    elif sche=='step':
        scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    else:
        scheduler = StepLR(optimizer, step_size=sche_cycle, gamma=0.1)

    return optimizer,scheduler, model

