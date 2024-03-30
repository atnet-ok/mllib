from src.config import *

@dataclass
class config:
    model: model_cfg = model_cfg()
    dataset: dataset_cfg = dataset_cfg()
    train: train_cfg = train_cfg()
    logger: logger_cfg =logger_cfg()