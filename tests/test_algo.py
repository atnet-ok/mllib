import unittest
from mllib.config.config import * 
from mllib.src.algo import *
from mllib.src.dataset import *

#python -m unittest tests.test_algo

class TestSimpleDeepLerning(unittest.TestCase):
    @unittest.skip('skipped')
    def test_init(self):
        cfg = get_config()
        algo = get_algo(cfg)

    @unittest.skip('skipped')
    def test_update(self):
        cfg = get_config("tests/data/default.json")
        algo = get_algo(cfg)
        train_loader, eval_loader = get_dataloader(cfg)
        algo.update(train_loader)

    def test_train(self):
        cfg = get_config("tests/data/default.json")
        algo = get_algo(cfg)
        train_loader, eval_loader = get_dataloader(cfg)
        algo.train( train_loader, eval_loader)

if __name__ == '__main__':
    unittest.main()