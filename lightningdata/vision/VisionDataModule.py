from pytorch_lightning.core.datamodule import LightningDataModule
import os
from torch.utils.data import DataLoader
import albumentations as A


class VisionDataModule(LightningDataModule):

    def __init__(self,
                 name: str = None,
                 num_workers: int = int(os.environ.get("NUM_WORKERS", 0)),
                 train_augment_policy: str = None,
                 batch_size: int = 4,
                 dataset_split_seed: int = 42,
                 patch_height: int = 512,
                 patch_width=512,
                 data_dir=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert name is not None
        self.name = name

        if not data_dir:
            data_dir = os.environ.get("OVERRIDE_DATA_DIR",
                                      os.environ["DATA_DIR"])
        if not data_dir:
            raise ValueError(
                "Must set data_dir, either through command line or DATA_DIR environment variable"
            )
        self.data_dir = os.path.join(os.path.abspath(data_dir), self.name)
        os.makedirs(self.data_dir, exist_ok=True)
        self.num_workers = int(num_workers)
        self.batch_size = batch_size
        self.dataset_split_seed = dataset_split_seed
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.train_augment_policy = train_augment_policy

    def get_augments(self):
        data_format = self.train_augment_policy[:-4]
        assert data_format in ["json", "yaml"]
        return A.load(self.train_augment_policy, data_format=data_format)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=2,
                          num_workers=self.num_workers,
                          shuffle=False,
                          drop_last=False,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          num_workers=self.num_workers,
                          shuffle=False,
                          drop_last=False,
                          pin_memory=True)
