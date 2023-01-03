import subprocess

import unittest
from mllib.src.config import * 
from mllib.src.trainer import *
from mllib.src.data import *
from mllib.src.logger import *

# python -m unittest tests.test_run
class TestRun(unittest.TestCase):
    @unittest.skip('skipped')
    def test_test_sklearn(self):
        command = 'python mllib/run.py -id sklearn_train -m train -model tests/data/model -cfg tests/data/config'.split(' ')
        subprocess.run(command)
        command = 'python mllib/run.py -id sklearn_test -m test -cfg tests/data/config'.split(' ')
        subprocess.run(command)

    def test_test_deep(self):
        command = 'python mllib/run.py -id deep_train -m train -model tests/data/model -cfg tests/data/config'.split(' ')
        subprocess.run(command)
        command = 'python mllib/run.py -id deep_test -m test -cfg tests/data/config'.split(' ')
        subprocess.run(command)