import unittest
from dllib.config import trainer_cfg
from dllib.app.trainer import MixupTrainer

# python -m unittest tests.test_trainer

class TestAll(unittest.TestCase):

    # def __init__(self, methodName: str = "runTest") -> None:
    #     super().__init__(methodName)
    #     self.cfg = None
    #     self.logger = None

    def setUp(self):
        pass

    def tearDown(self):
        pass

    #@unittest.skip('skipped')
    def test_trainer(self):

        cfg_trainer = trainer_cfg()

        trainer = MixupTrainer(cfg_trainer)
        trainer.train()


        
if __name__ == '__main__':
    unittest.main()