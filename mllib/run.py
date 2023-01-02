from mllib.src.config import *
from mllib.src.algo import *
from mllib.src.dataset import *
from mllib.src.algo import *
from mllib.src.model import *

if __name__=='__main__':
    config_path="mllib/config/default.yaml"
    cfg = get_config(
        config_path=config_path
        )
    train_loader, eval_loader = get_dataloader(cfg)
    algo = get_algo(cfg)
    algo.train(train_loader)
