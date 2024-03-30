import unittest
from src.config import dataset_cfg
# from src.manager import *
# from src.app.trainer import *
# python -m unittest tests.test_dataset

class TestAll(unittest.TestCase):

    # def __init__(self, methodName: str = "runTest") -> None:
    #     super().__init__(methodName)
    #     self.cfg = None
    #     self.logger = None

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # def test_cfg(self):
    #     cfg = get_config("config/000_default.yaml")
    #     self.assertEqual(cfg.model.name, 'tf_efficientnet_b7')

    #@unittest.skip('skipped')
    def test_dataset(self):
        dataset_cfg = dataset_cfg()
        dataset_train = get_dataset(cfg, phase='train')
        dataset_eval  = get_dataset(cfg, phase='eval')
        print(dataset_train.__len__())
        print(dataset_train.__getitem__(0))
        print(dataset_eval.__len__())
        print(dataset_eval.__getitem__(0))

        dl_train = get_dataloader(self.cfg, 'train',dataset_train)
        dl_eval = get_dataloader(self.cfg, 'eval',dataset_eval)

    # @unittest.skip('skipped')
    # def test_model(self):
    #     cfg = get_config("config/000_default.yaml")
    #     model = get_model(cfg)
    #     print(model)

    # @unittest.skip('skipped')
    # def test_trainer(self):
    #     class args:
    #          experiment_name = "000_test"
    #          run_name = '000_default'
    #          mode = 'train'
    #          cfg_dir = "config/"
    #          model_dir = "model/"
    #          log_dir = "log/"
    #     manager = Manager(args)
    #     cfg, logger = manager.set_experiment()

    #     trainer = get_trainer(cfg, logger)
    #     trainer.train()
 


if __name__ == '__main__':
    unittest.main()