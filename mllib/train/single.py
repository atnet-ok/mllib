from mllib.config.config import *
from mllib.src.algo import *
from mllib.src.dataset import *
from mllib.src.algo import *
from mllib.src.model import *

cfg = get_config(
    config_path="mllib/config/default.json"
    )

train_loader, eval_loader = get_dataloader(cfg)
algo = get_algo(cfg)

algo.train(train_loader)




