import unittest
from mllib.config.config import * 
from mllib.src.model import * 

class TestModel(unittest.TestCase):
    def test_model(self):
        cfg = get_config("tests/data/default.yaml")
        model = get_model(cfg)
        # self.assertEqual(model, cfg.model.name)

if __name__ == '__main__':
    unittest.main()