from ..VisionDataModule import VisionDataModule
from torchvision import datasets
import os
from functools import partial
import torch
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY

__all__ = ["MNist"]


def albumentations_transform(image, transform):
    return transform(image=image)


class MNistDataSet(datasets.MNIST):

    def __init__(self,
                 root,
                 transform=None,
                 split="train",
                 val_percentage=0.10,
                 seed=42):

        logging.basicConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.task = "classification"

        assert split in ["train", "val", "test"]

        self.logger.info(f"Loading {split} split from {root}")

        if transform is not None:
            transform = partial(albumentations_transform, transform=transform)

        super().__init__(root,
                         transform=transform,
                         train=split in ["train", "val"],
                         download=True)

        self.patch_transform = A.Compose([
            A.PadIfNeeded(min_height=32, min_width=32, always_apply=True),
            A.RandomCrop(height=32, width=32, always_apply=True)
        ])

        if split in ["train", "val"]:
            val_count = int(len(self.data) * val_percentage)
            g = torch.Generator()
            g.manual_seed(seed)
            trainval_indeces = torch.randperm(len(self.data), generator=g)
            val_indices = trainval_indeces[:val_count]
            train_indeces = trainval_indeces[val_count:]

            if split == "train":
                self.data = self.data[train_indeces]
                self.targets = self.targets[train_indeces]
            else:
                self.data = self.data[val_indices]
                self.targets = self.targets[val_indices]

        self.targets = torch.stack([
            torch.nn.functional.one_hot(t, num_classes=len(self.classes))
            for t in self.targets
        ])

        self.data = self.data.numpy()
        self.targets = self.targets.numpy()

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        if self.patch_transform is not None:
            img = self.patch_transform(image=img)["image"]

        expected_shape = img.shape

        if self.transform is not None:
            img = self.transform(img)["image"]

        if self.patch_transform is not None and expected_shape != img.shape:
            img = self.patch_transform(image=img)["image"]

        img = A.ToFloat(always_apply=True)(image=img)["image"]
        img = ToTensorV2(always_apply=True)(image=img)["image"]

        target = torch.from_numpy(target).to(img.dtype)

        return img, target

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "processed")


@DATAMODULE_REGISTRY
class MNist(VisionDataModule):

    Dataset = MNistDataSet
