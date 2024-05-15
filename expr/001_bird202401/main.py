import sys
import pathlib

module_path = pathlib.Path(__file__, "..", "../..").resolve()
if module_path not in sys.path:
    sys.path.append(str(module_path))

import hydra
from omegaconf import OmegaConf
from dataclasses import dataclass

from dllib.config import trainer_cfg,logger_cfg
from dllib.app.trainer import MixupTrainer
from dllib.common.utils import date2str

@dataclass
class config:
    trainer:trainer_cfg=trainer_cfg()
    logger:logger_cfg=logger_cfg()

@hydra.main(config_name="config", version_base=None, config_path="config")
def main(cfg:config) -> None:

    cfg.logger.experiment_name = "birdclef2024"
    cfg.logger.run_name = date2str()

    print(OmegaConf.to_yaml(cfg))

    cfg_trainer = trainer_cfg()

    trainer = MixupTrainer(cfg_trainer)
    trainer.train()

if __name__ == "__main__":

    cfg = config()
    OmegaConf.save(cfg, 'config/config.yaml')
    main()