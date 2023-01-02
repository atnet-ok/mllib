import torch.optim as optim

def get_optimizer(cfg,model):
    opt = cfg.train.optimizer
    lr = cfg.train.lr
    weight_decay = cfg.train.wd

    if opt == 'sgd':
        #optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])
        optimizer = optim.SGD(
            model.parameters(), 
            lr=lr , 
            momentum=0, #0.9, 
            weight_decay=weight_decay #1e-4
            )
    elif opt == 'adam':
        optimizer = optim.Adam(
                                model.parameters(), 
                                lr=lr ,
                                #eps=cfg['eps'],
                                #betas=(cfg['b0'],cfg['b1'])
                                )
    elif opt == 'radam':
        optimizer = optim.RAdam(model.parameters(), 
                                lr=lr , 
                                betas=(0.9, 0.999), 
                                eps=1e-08, 
                                weight_decay=0,
                                )
    elif opt == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    elif opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=cfg['lr'], alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

    return optimizer, model

def get_scheduler(cfg,optimizer):
    sche =  cfg['scheduler']
    
    if sche=='cosine_warmup':
        from timm.scheduler import CosineLRScheduler
        scheduler = CosineLRScheduler(
                                    optimizer, 
                                    t_initial=cfg['epoch'], 
                                    lr_min=1e-5, 
                                    warmup_t=5, 
                                    warmup_lr_init=1e-5, 
                                    warmup_prefix=True
                                    )
        # scheduler = CosineLRScheduler(optimizer, t_initial=cfg['epoch'], lr_min=cfg['lr_min'], 
        #                     warmup_t=3, warmup_lr_init=cfg['lr_init'], warmup_prefix=True)
    elif sche=='cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-4, T_max= cfg['epoch'])
    elif sche=='step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif sche=='none':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['epoch'], gamma=0.1)

    return scheduler, optimizer