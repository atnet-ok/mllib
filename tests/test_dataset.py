import unittest
from dllib.config import dataset_cfg, dataloader_cfg
from dllib.domain.dataset import get_dataset, get_dataloader

# python -m unittest tests.test_dataset

class TestDataset(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    #@unittest.skip('skipped')
    def test_dataset(self):

        cfg_dataset = dataset_cfg()
        cfg_dataloader = dataloader_cfg()

        for phase in ["train","eval"]:
            dataset = get_dataset(cfg_dataset,phase)
            print(dataset.__len__())
            output_ = dataset.__getitem__(0)
            print(output_  )

            dataloader = get_dataloader(dataset,cfg_dataloader,phase)
            for x,y in dataloader:
                print(x.shape)
                print(y)
                break


if __name__ == '__main__':
    unittest.main()