from torchvision import datasets
from lightningdata.vision import VisionDataModule
import logging


class MNIST(VisionDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(name="mnist", *args, **kwargs)

        self.logger = logging.getLogger(__name__)

        try:
            datasets.MNIST(root=self.data_dir)
        except RuntimeError as e:
            if "Dataset not found" in str(e):
                self.logger.warn(e)
            else:
                raise e

    def initialize(self):
        datasets.MNIST(root=self.data_dir, download=True)

    @property
    def train_data(self):
        return datasets.MNIST(root=self.data_dir, train=True).data

    @property
    def test_data(self):
        return datasets.MNIST(root=self.data_dir, train=False).data
