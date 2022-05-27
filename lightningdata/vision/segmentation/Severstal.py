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

    task = "segmentation"

    def __init__(self, seed=42, *args, **kwargs):
        kwargs["name"] = "severstal"
        super().__init__(*args, **kwargs)

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
