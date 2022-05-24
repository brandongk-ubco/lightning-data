import os
import glob
import numpy as np
from functools import partial
import random
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import PIL

def albumentations_transform(image, transform):
    return transform(image=image)

class FolderDataset(datasets.VisionDataset):
    def __init__(self,
                 root,
                 transform=None,
                 val_percentage=0.10,
                 test_percentage=0.10,
                 split="train",
                 extensions=["tiff", "tif"],
                 patch_height=32,
                 patch_width=32,
                 seed=42):

        self.split = split

        assert split in ["all", "train", "val", "test"]

        if transform is not None:
            transform = partial(albumentations_transform, transform=transform)

        self.patch_transform = A.Compose([
            A.PadIfNeeded(min_height=patch_height,
                          min_width=patch_width,
                          border_mode=0,
                          always_apply=True),
            A.RandomCrop(height=patch_height,
                         width=patch_width,
                         always_apply=True)
        ])

        super().__init__(root, transform=transform)

        self.data = []
        for extension in extensions:
            file_search = os.path.join(self.root, "*", f"*.{extension}")
            files = glob.glob(file_search)
            self.data += files

        self.classes = list(set([os.path.split(os.path.dirname(f))[-1] for f in files]))

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
            self.data = self.data[train_count + val_count:train_count +
                                  val_count + test_count]

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = PIL.Image.open(image_path)
        image_class = os.path.split(os.path.dirname(image_path))[-1]
        image = np.array(image)
        if image.shape[2] == 4:
            assert image[:, :, 3].min() == 255
            image = image[:, :, :3]
        image = image.astype(np.float32) / 255.
        targets = np.zeros(len(self.classes), dtype=np.float32)
        target_idx = self.classes.index(image_class)
        targets[target_idx] = 1.

        if self.split == "train":
            image = self.augment(image)
        else:
            image = self.val_augment(image)

        image = A.ToFloat(always_apply=True)(image=image)["image"]
        image = ToTensorV2(always_apply=True)(image=image)["image"]

        targets = torch.from_numpy(targets).to(image.dtype)

        return image, targets

    def val_augment(self, image):
        if self.patch_transform is not None:
            transformed = self.patch_transform(image=image)
            image = transformed["image"]
        return image

    def augment(self, image):
        if self.patch_transform is not None:
            transformed = self.patch_transform(image=image)
            image = transformed["image"]

        expected_shape = image.shape

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        if self.patch_transform is not None and expected_shape != image.shape:
            transformed = self.patch_transform(image=image)
            image = transformed["image"]

        return image

    def __len__(self):
        return len(self.data)
