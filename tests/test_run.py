import subprocess

import unittest
from mllib.src.config import * 
from mllib.src.trainer import *
from mllib.src.data import *
from mllib.src.logger import *

# python -m unittest tests.test_run
class TestRun(unittest.TestCase):

    #@unittest.skip('skipped')
    def test_test_sklearn(self):
        command = 'python mllib/run.py -run sklearn_train -m train -model tests/data/model -cfg tests/data/config -log tests/data/log'
        subprocess.run(command.split(' '))
        # command = 'python mllib/run.py -run sklearn_test -m test -cfg tests/data/config -log tests/data/log'
        # subprocess.run(command.split(' '))
    @unittest.skip('skipped')
    def test_test_deep(self):
        command = 'python mllib/run.py -run deep_train -m train -model tests/data/model -cfg tests/data/config -log tests/data/log'
        subprocess.run(command.split(' '))
        # command = 'python mllib/run.py -run deep_test -m test -cfg tests/data/config -log tests/data/log'
        # subprocess.run(command.split(' '))