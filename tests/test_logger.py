import unittest
from mllib.src.logger import * 
from mllib.src.config import *
from mllib.src.utils import *

# python -m unittest tests.test_logger

class TestLogger(unittest.TestCase):
    def test_default(self):
        class args:
             experiment_name = "test"
             run_name = '000_default'
             mode = 'train'
             cfg_dir = "config/"
             model_dir = "model/"
             log_dir = "log/"

        cfg, logger = start_experiment(args)


        end_experiment(args,  None, None)