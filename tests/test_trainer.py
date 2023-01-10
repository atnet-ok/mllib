import unittest
from mllib.src.config import * 
from mllib.src.trainer import *
from mllib.src.data import *
from mllib.src.logger import *

# python -m unittest tests.test_trainer.TestDLTrainer
# python -m unittest tests.test_trainer.TestMLTrainer

class TestDLTrainer(unittest.TestCase):
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

    @unittest.skip('skipped')
    def test_officehome(self):
        cfg = get_config("tests/data/config/officehome.yaml")
        trainer = get_trainer(cfg)
        trainer.train()

class TestMLTrainer(unittest.TestCase):
    def setUp(self):
        class args:
             experiment_name = "test"
             run_name = 'sklearn_train'
             mode = 'train'
             cfg_dir = "tests/data/config/"
             model_dir = "tests/data/model/"
             log_dir = "tests/data/log/"

        self.args = args

    @unittest.skip('skipped')
    def test_sklearn(self):

        model_s = [
            "RandomForestClassifier",
            #"SVC",
            #"GradientBoostingClassifier",
            #"LogisticRegression"
            ]

        cfg, logger = start_experiment(self.args)
        for model in model_s:
            cfg.model.name = model
            trainer = get_trainer(cfg,logger)
            _ = trainer.train()
            _ = trainer.test()

    # @unittest.skip('skipped')
    def test_mldatrainer(self):
        self.args.run_name = 'sklearn_train_da'
        cfg, logger = start_experiment(self.args)
        trainer = get_trainer(cfg,logger)
        _ = trainer.train()

if __name__ == '__main__':
    unittest.main()