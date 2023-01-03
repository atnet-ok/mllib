import subprocess

import unittest
from mllib.src.config import * 
from mllib.src.trainer import *
from mllib.src.data import *
from mllib.src.logger import *

# python -m unittest tests.test_run
class TestRun(unittest.TestCase):
    # @unittest.skip('skipped')
    def test_run(self):
        command = 'python mllib/run.py -id sklearn -cfg tests/data'.split(' ')
        subprocess.run(command)