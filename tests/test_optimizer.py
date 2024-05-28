import unittest
from dllib.config import optimizer_cfg
from dllib.common.optimizer import get_optimizer
import matplotlib.pyplot as plt
from torch import nn

# python -m unittest tests.test_optimizer

class TestAll(unittest.TestCase):

    # def __init__(self, methodName: str = "runTest") -> None:
    #     super().__init__(methodName)
    #     self.cfg = None
    #     self.logger = None

    def setUp(self):
        pass

    def tearDown(self):
        pass

    #@unittest.skip('skipped')
    def test_optimizer(self):

        cfg = optimizer_cfg()

        model = nn.Linear(12 * 12 * 64, 128)

        optimizer, scheduler, model = get_optimizer(cfg,model)

        lrs = []
        for epoch in range(1, 100):
            scheduler.step(epoch)
            lrs.append(optimizer.param_groups[0]["lr"])

        plt.plot(range(len(lrs)), lrs)
        plt.savefig(".tmp/lr_hist")

        
if __name__ == '__main__':
    unittest.main()