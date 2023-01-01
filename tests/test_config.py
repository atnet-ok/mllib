import unittest
from mllib.config.config import * 

#python -m unittest tests.test_config

class TestConfig(unittest.TestCase):
    def test_config(self):
        cfg = get_config("tests/data/default.json")
        self.assertEqual(cfg.model.name, 'resnet50')

if __name__ == '__main__':
    unittest.main()