from .CompressedNpzDataset import CompressedNpzDataset
from ..VisionDataModule import VisionDataModule
import os
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
import albumentations as A


class SeverstalDataset(CompressedNpzDataset):

    classes = ["1", "2", "3", "4"]
    in_channels = 1

    def __init__(self, patch_height=256, patch_width=256, *args, **kwargs):
        super().__init__(patch_height=patch_height,
                         patch_width=patch_width,
                         *args,
                         **kwargs)


@DATAMODULE_REGISTRY
class Severstal(VisionDataModule):

    def __init__(self, seed=42, augment_policy_path=None, *args, **kwargs):
        kwargs["name"] = "severstal"
        super().__init__(*args, **kwargs)

        self.task = "segmentation"

        self.seed = seed
        self.train_dataset = SeverstalDataset(root=self.data_dir,
                                              split="train",
                                              transform=self.augments,
                                              seed=self.seed)

        self.val_dataset = SeverstalDataset(root=self.data_dir,
                                            split="val",
                                            seed=self.seed)

        self.test_dataset = SeverstalDataset(root=self.data_dir,
                                             split="test",
                                             seed=self.seed)

        self.all_dataset = SeverstalDataset(root=self.data_dir,
                                            split="all",
                                            seed=self.seed)

    @property
    def in_channels(self):
        return self.train_dataset.in_channels

    @property
    def num_classes(self):
        return len(self.train_dataset.classes)

    @property
    def classes(self):
        return self.train_dataset.classes

    @property
    def train_data(self):
        return self.train_dataset.data

    @property
    def val_data(self):
        return self.val_dataset.data

    @property
    def test_data(self):
        return self.test_dataset.data

    @property
    def all_data(self):
        return self.all_dataset.data
