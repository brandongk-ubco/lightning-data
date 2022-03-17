from lightningdata.vision.classification import MNistDataModule
from enum import Enum


class Datasets(Enum):
    mnist = "mnist"

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @classmethod
    def choices(cls):
        return sorted([e.value for e in cls])

    def get(dataset):
        if dataset == "mnist":
            return MNistDataModule

        raise ValueError("Dataset {} not defined".format(dataset))
