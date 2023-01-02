import unittest
from mllib.src.config import * 
from mllib.src.data import * 

#python -m unittest tests.test_dataset
class TestDataset(unittest.TestCase):
    def test_dataloader(self):
        cfg = get_config("tests/data/default.yaml")
        train_loader, eval_loader  = get_dataloader(cfg)
        print(train_loader.dataset)
        print(len(train_loader.dataset))
        for data,label in train_loader:
            print(data.shape)
            print(label.shape)
            print(label)
            break
        for data,label in eval_loader:
            print(data.shape)
            print(label.shape)
            print(label)
            break
        # self.assertEqual(model, cfg.model.name)

if __name__ == '__main__':
    unittest.main()