from dllib.config import *

@dataclass
class config:
    model: model_cfg = model_cfg()
    dataset: dataset_cfg = dataset_cfg()
    train: trainer_cfg = trainer_cfg()
    logger: logger_cfg =logger_cfg()