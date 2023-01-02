import unittest
from mllib.src.config import * 

# python -m unittest tests.test_config

class TestConfig(unittest.TestCase):
    def test_default(self):
        cfg = get_config("tests/data/default.yaml")
        self.assertEqual(cfg.model.name, 'resnet50')

    def test_crwu(self):
        cfg = get_config("mllib/config/CWRU.yaml")
        self.assertEqual(cfg.data.name, 'CWRUsyn2real')

if __name__ == '__main__':
    unittest.main()