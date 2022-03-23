from pytorch_lightning.core.datamodule import LightningDataModule
import os
from torch.utils.data import DataLoader
import albumentations as A


class VisionDataModule(LightningDataModule):

    def __init__(self,
                 name: str = None,
                 num_workers: int = int(os.environ.get("NUM_WORKERS", 0)),
                 augment_policy_path: str = None,
                 batch_size: int = 4,
                 dataset_split_seed: int = 42,
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

        import pdb
        pdb.set_trace()

        if augment_policy_path is None:
            augment_policy_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "policies",
                f"{self.name}.yaml")

        assert os.path.exists(augment_policy_path)
        augment_data_format = augment_policy_path[-4:]
        assert augment_data_format in ["yaml", "json"]
        self.augments = A.load(augment_policy_path,
                               data_format=augment_data_format)

        os.makedirs(self.data_dir, exist_ok=True)
        self.num_workers = int(num_workers)
        self.batch_size = batch_size
        self.dataset_split_seed = dataset_split_seed

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=1,
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

    def all_dataloader(self):
        return DataLoader(self.all_dataset,
                          batch_size=1,
                          num_workers=self.num_workers,
                          shuffle=False,
                          drop_last=False,
                          pin_memory=True)