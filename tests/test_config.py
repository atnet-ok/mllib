import unittest
from mllib.src.config import * 

#python -m unittest tests.test_config

class TestConfig(unittest.TestCase):
    def test_config(self):
        cfg = get_config("tests/data/default.yaml")
        self.assertEqual(cfg.model.name, 'resnet50')

if __name__ == '__main__':
    unittest.main()