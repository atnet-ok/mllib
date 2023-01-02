import unittest
from mllib.src.config import * 
from mllib.src.trainer import *
from mllib.src.dataset import *

#python -m unittest tests.test_trainer

class TestSimpleDeepLerning(unittest.TestCase):
    @unittest.skip('skipped')
    def test_init(self):
        cfg = get_config()
        trainer = get_trainer(cfg)

    @unittest.skip('skipped')
    def test_update(self):
        cfg = get_config("tests/data/default.yaml")
        trainer = get_trainer(cfg)
        trainer.update()

    def test_train(self):
        cfg = get_config("tests/data/default.yaml")
        trainer = get_trainer(cfg)
        trainer.train()

if __name__ == '__main__':
    unittest.main()