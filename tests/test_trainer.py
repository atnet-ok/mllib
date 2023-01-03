import unittest
from mllib.src.config import * 
from mllib.src.trainer import *
from mllib.src.data import *
from mllib.src.logger import *

# python -m unittest tests.test_trainer.TestDeepLerning
# python -m unittest tests.test_trainer.TestSKLearn

class TestDeepLerning(unittest.TestCase):
    @unittest.skip('skipped')
    def test_init(self):
        cfg = get_config()
        trainer = get_trainer(cfg)

    @unittest.skip('skipped')
    def test_update(self):
        cfg = get_config("tests/data/config/default.yaml")
        trainer = get_trainer(cfg)
        trainer.update()

    @unittest.skip('skipped')
    def test_default(self):
        cfg = get_config("tests/data/config/default.yaml")
        trainer = get_trainer(cfg)
        trainer.train()

    @unittest.skip('skipped')
    def test_cwru(self):
        cfg = get_config("tests/data/config/cwru.yaml")
        trainer = get_trainer(cfg)
        trainer.train()

    def test_officehome(self):
        cfg = get_config("tests/data/config/officehome.yaml")
        trainer = get_trainer(cfg)
        trainer.train()

class TestSKLearn(unittest.TestCase):
    def test_sklearn(self):
        class args:
             experiment_name = "test"
             run_id = 'sklearn_train'
             mode = 'train'
             cfg_dir = "tests/data/config/"
             model_dir = "tests/data/model/"
             log_dir = "tests/data/log/"
        model_s = [
            "RandomForestClassifier",
            #"SVC",
            #"GradientBoostingClassifier",
            #"LogisticRegression"
            ]

        cfg, logger = start_experiment(args)
        for model in model_s:
            cfg.model.name = model
            trainer = get_trainer(cfg,logger)
            _ = trainer.train()
            _ = trainer.test()

if __name__ == '__main__':
    unittest.main()