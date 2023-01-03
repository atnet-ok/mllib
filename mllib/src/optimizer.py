import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR
from timm.scheduler import CosineLRScheduler

def get_optimizer(cfg,model):
    opt = cfg.train.optimizer
    lr = cfg.train.lr
    weight_decay = cfg.train.wd
    momentum = cfg.train.momentum

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

    return optimizer, model

def get_scheduler(cfg,optimizer):
    sche =  cfg.train.scheduler
    epoch =cfg.train.epoch
    lr = cfg.train.lr
    
    if sche=='cosine_warmup':
        scheduler = CosineLRScheduler(
                optimizer, 
                t_initial=epoch , 
                lr_min=lr*1e-1, 
                warmup_t=5, 
                warmup_lr_init=lr*1e-1, 
                warmup_prefix=True
                )
    elif sche=='cosine':
        scheduler = CosineAnnealingLR(
            optimizer, 
            eta_min=1e-4, 
            T_max= epoch)
    elif sche=='step':
        scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    elif sche=='none':
        scheduler = StepLR(optimizer, step_size=epoch, gamma=0.1)

    return scheduler, optimizer