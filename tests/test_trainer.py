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
        cfg = get_config("tests/data/default.yaml")
        trainer = get_trainer(cfg)
        trainer.update()

    @unittest.skip('skipped')
    def test_default(self):
        cfg = get_config("tests/data/default.yaml")
        trainer = get_trainer(cfg)
        trainer.train()

    @unittest.skip('skipped')
    def test_cwru(self):
        cfg = get_config("tests/data/cwru.yaml")
        trainer = get_trainer(cfg)
        trainer.train()

    def test_officehome(self):
        cfg = get_config("tests/data/officehome.yaml")
        trainer = get_trainer(cfg)
        trainer.train()

class TestSKLearn(unittest.TestCase):
    def test_sklearn(self):
        model_s = [
            "RandomForestClassifier",
            #"SVC",
            #"GradientBoostingClassifier",
            #"LogisticRegression"
            ]
        cfg = get_config("tests/data/sklearn.yaml")
        logger = start_logging()

        for model in model_s:
            cfg.model.name = model
            trainer = get_trainer(cfg,logger)
            trainer.train()
            trainer.test()



if __name__ == '__main__':
    unittest.main()