import unittest
from dllib.config import dataset_cfg, dataloader_cfg
from dllib.domain.dataset import get_dataset, get_dataloader

# python -m unittest tests.test_dataset

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
    def test_dataset(self):

        cfg_dataset = dataset_cfg()
        cfg_dataloader = dataloader_cfg()

        for phase in ["train","eval"]:
            dataset = get_dataset(cfg_dataset,phase)
            print(dataset.__len__())
            output_ = dataset.__getitem__(0)
            print(output_  )

            print(dataset.target_columns)
            print(dataset.bird2id)

            # dataloader = get_dataloader(dataset,cfg_dataloader,phase)
            # for x,y in dataloader:
            #     print(x.shape)
            #     print(y)
            #     break

    def test_test(self):

        phase = "test"
        cfg_dataset = dataset_cfg()
        dataset = get_dataset(cfg_dataset,phase)

        print(dataset.__len__())

        for i in range(dataset.__len__()):
            img = dataset.__getitem__(i)
            print(img.shape)
            print(img)


if __name__ == '__main__':
    unittest.main()