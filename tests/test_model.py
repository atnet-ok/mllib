import unittest
import torch
from dllib.config import model_cfg
from dllib.domain.model import get_model

# python -m unittest tests.test_model


class TestModel(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    # @unittest.skip('skipped')
    def test_model(self):
        cfg_model = model_cfg()

        model = get_model(cfg_model)
        x = torch.rand(32, 1, 28, 28)
        y = model(x)
        print(y.shape)


if __name__ == "__main__":
    unittest.main()
