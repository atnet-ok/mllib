from mllib.src.config import *
from mllib.src.trainer import *

if __name__=='__main__':
    config_path="mllib/config/default.yaml"
    cfg = get_config(
        config_path=config_path
        )
    algo = get_trainer(cfg)
    algo.train()
    algo.test()
