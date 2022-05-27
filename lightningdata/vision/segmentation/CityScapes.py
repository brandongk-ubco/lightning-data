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

    task = "segmentation"

    def __init__(self, seed=42, *args, **kwargs):
        kwargs["name"] = "cityscapes"
        super().__init__(*args, **kwargs)

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
