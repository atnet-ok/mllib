import unittest
from dllib.config import trainer_cfg
from dllib.app.trainer import get_trainer

# python -m unittest tests.test_trainer


class TestTrainer(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    # @unittest.skip('skipped')
    def test_trainer(self):
        cfg_trainer = trainer_cfg()
        trainer = get_trainer(cfg_trainer)
        trainer.train()


if __name__ == "__main__":
    unittest.main()
