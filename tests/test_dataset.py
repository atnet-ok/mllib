import unittest
from mllib.src.config import * 
from mllib.src.data import * 

# python -m unittest tests.test_dataset
class TestDataset(unittest.TestCase):
    @unittest.skip('skipped')
    def test_default(self):
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

    @unittest.skip('skipped')
    def test_cwru(self):
        cfg = get_config("tests/data/CWRU.yaml")
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

    def test_officehome(self):
        cfg = get_config("tests/data/officehome.yaml")
        dataset_train, dataset_eval  = get_dataset(cfg)
        print(dataset_train.__len__())
        print(dataset_train.__getitem__(0))
        print(dataset_eval.__len__())
        print(dataset_eval.__getitem__(0))

if __name__ == '__main__':
    unittest.main()