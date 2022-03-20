import os
import glob
import numpy as np
from functools import partial
import random
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch


def albumentations_transform(image, mask, transform):
    return transform(image=image, mask=mask)


class CompressedNpzDataset(datasets.VisionDataset):

    def __init__(self,
                 root,
                 transform=None,
                 val_percentage=0.10,
                 test_percentage=0.10,
                 split="train",
                 patch_height=32,
                 patch_width=32,
                 seed=42):

        self.split = split

        assert split in ["train", "val", "test"]

        if transform is not None:
            transform = partial(albumentations_transform, transform=transform)

        self.patch_transform = A.Compose([
            A.PadIfNeeded(min_height=patch_height,
                          min_width=patch_width,
                          always_apply=True),
            A.RandomCrop(height=patch_height,
                         width=patch_width,
                         always_apply=True)
        ])

        super().__init__(root, transform=transform)

        file_search = os.path.join(self.root, "*.npz")
        files = glob.glob(file_search)
        self.data = [os.path.splitext(os.path.basename(f))[0] for f in files]
        random.Random(seed).shuffle(self.data)

        test_count = int(len(self.data) * test_percentage)
        trainval_count = len(self.data) - test_count
        val_count = int(trainval_count * val_percentage)
        train_count = trainval_count - val_count

        if split == "train":
            self.data = self.data[:train_count]
        elif split == "val":
            self.data = self.data[train_count:train_count + val_count]
        elif split == "test":
            self.data = self.data[:train_count + val_count:train_count +
                                  val_count + test_count]
        else:
            raise ValueError("Split must be train, val, or test.")

    def __getitem__(self, idx):
        image_name = self.data[idx]
        image_file = os.path.join(self.root, "{}.npz".format(image_name))
        image_np = np.load(image_file)

        img = image_np["image"]
        target = image_np["mask"]

        if self.patch_transform is not None:
            transformed = self.patch_transform(image=img, mask=target)
            img = transformed["image"]
            target = transformed["mask"]

        expected_shape = img.shape

        if self.transform is not None:
            transformed = self.transform(image=img, mask=target)
            img = transformed["image"]
            target = transformed["mask"]

        if self.patch_transform is not None and expected_shape != img.shape:
            transformed = self.patch_transform(image=img, mask=target)
            img = transformed["image"]
            target = transformed["mask"]

        img = A.ToFloat(always_apply=True)(image=img)["image"]
        img = ToTensorV2(always_apply=True)(image=img)["image"]

        target = torch.from_numpy(target).to(img.dtype).moveaxis(2, 0)

        return img, target

    def __len__(self):
        return len(self.data)
