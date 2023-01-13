import unittest
import mlflow
from mllib.src.config import * 
from mllib.src.trainer import *
from mllib.src.data import *
from mllib.src.logger import *

# python -m unittest tests.test_trainer.TestDLTrainer
# python -m unittest tests.test_trainer.TestMLTrainer

class TestDLTrainer(unittest.TestCase):
    def setUp(self):
        class args:
             experiment_name = "test"
             run_name = 'sklearn_train'
             mode = 'train'
             cfg_dir = "tests/data/config/"
             model_dir = "tests/data/model/"
             log_dir = "tests/data/log/"

        self.args = args
        print('Called `setUp`.')

    def tearDown(self):
        mlflow.end_run()

    #@unittest.skip('skipped')
    def test_default(self):
        args = self.args
        args.run_name = 'default'
        cfg, logger = start_experiment(args)
        trainer = get_trainer(cfg,logger)
        trainer.train()

    @unittest.skip('skipped')
    def test_cwru(self):
        args = self.args
        args.run_name = 'cwru'
        cfg, logger = start_experiment(args)
        trainer = get_trainer(cfg,logger)
        trainer.train()

    @unittest.skip('skipped')
    def test_officehome(self):
        args = self.args
        args.run_name = 'officehome'
        cfg, logger = start_experiment(args)
        trainer = get_trainer(cfg,logger)
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

    def tearDown(self):
        mlflow.end_run()

    @unittest.skip('skipped')
    def test_sklearn(self):
        args = self.args
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

    #@unittest.skip('skipped')
    def test_mldatrainer(self):
        args = self.args
        args.run_name = 'sklearn_train_da'
        cfg, logger = start_experiment(args)
        cfg.train.seed = 0
        trainer = get_trainer(cfg,logger)
        _ = trainer.train()

    @unittest.skip('skipped')
    def test_fewrealtrainer(self):
        
        cfg, logger = start_experiment(self.args)
        cfg.data.eval_rate = 0.996
        cfg.train.seed = 0
        trainer = get_trainer(cfg,logger)
        _ = trainer.train()

if __name__ == '__main__':
    unittest.main()