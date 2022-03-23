from .CompressedNpzDataset import CompressedNpzDataset
from ..VisionDataModule import VisionDataModule
import os
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
import albumentations as A


class CityScapesDataset(CompressedNpzDataset):

    classes = [
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain", "sky",
        "person", "rider", "car", "truck", "bus", "train", "motorcycle",
        "bicycle"
    ]
    in_channels = 3

    def __init__(self, patch_height=384, patch_width=384, *args, **kwargs):
        super().__init__(patch_height=patch_height,
                         patch_width=patch_width,
                         *args,
                         **kwargs)

    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        target = target[1:, :, :]
        return image, target


@DATAMODULE_REGISTRY
class CityScapes(VisionDataModule):

    def __init__(self, seed=42, augment_policy_path=None, *args, **kwargs):
        kwargs["name"] = "cityscapes"
        super().__init__(*args, **kwargs)

        self.task = "segmentation"

        self.seed = seed
        self.train_dataset = CityScapesDataset(root=self.data_dir,
                                               split="train",
                                               transform=self.augments,
                                               seed=self.seed)

        self.val_dataset = CityScapesDataset(root=self.data_dir,
                                             split="val",
                                             seed=self.seed)

        self.test_dataset = CityScapesDataset(root=self.data_dir,
                                              split="test",
                                              seed=self.seed)

        self.all_dataset = CityScapesDataset(root=self.data_dir,
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
