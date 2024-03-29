from pytorch_lightning.core.datamodule import LightningDataModule
import os
from torch.utils.data import DataLoader
import albumentations as A
import multiprocessing
from torchvision.datasets import VisionDataset


class VisionDataModule(LightningDataModule):

    Dataset: VisionDataset

    def __init__(self,
                 num_workers: int = int(
                     os.environ.get("NUM_WORKERS",
                                    multiprocessing.cpu_count() // 2)),
                 augment_policy_path: str = None,
                 batch_size: int = 4,
                 seed: int = 42,
                 data_dir=None,
                 preprocessing="normalization",
                 *args,
                 **kwargs):

        details = self.__class__.__module__.split(".")
        self.dataset_name = details[-1]

        if not hasattr(self, 'category'):
            self.category = details[1]
        if not hasattr(self, 'task'):
            self.task = details[2]

        super().__init__(*args, **kwargs)

        if not data_dir:
            data_dir = os.environ.get("OVERRIDE_DATA_DIR",
                                      os.environ["DATA_DIR"])
        if not data_dir:
            raise ValueError(
                "Must set data_dir, either through command line or DATA_DIR environment variable"
            )
        self.data_dir = os.path.join(os.path.abspath(data_dir),
                                     self.dataset_name)

        if augment_policy_path is None:
            override_file = os.path.join(os.getcwd(), "Augments.yaml")

            if os.path.exists(override_file):
                augment_policy_path = override_file
            else:
                augment_policy_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "policies",
                    f"{self.dataset_name}.yaml")

        assert os.path.exists(augment_policy_path)
        augment_data_format = augment_policy_path[-4:]
        assert augment_data_format in ["yaml", "json"]
        self.augments = A.load(augment_policy_path,
                               data_format=augment_data_format)

        os.makedirs(self.data_dir, exist_ok=True)
        self.num_workers = int(num_workers)
        self.batch_size = batch_size
        self.seed = seed
        self.dataset_split_seed = self.seed

        self.train_dataset = self.Dataset(root=self.data_dir,
                                          split="train",
                                          preprocessing=preprocessing,
                                          transform=self.augments,
                                          seed=self.seed)

        self.val_dataset = self.Dataset(root=self.data_dir,
                                        split="val",
                                        preprocessing=preprocessing,
                                        seed=self.seed)

        self.test_dataset = self.Dataset(root=self.data_dir,
                                         split="test",
                                         preprocessing=preprocessing,
                                         seed=self.seed)

        self.predict_dataset = self.Dataset(root=self.data_dir,
                                            split="test",
                                            preprocessing=preprocessing,
                                            seed=self.seed)

        self.all_dataset = self.Dataset(root=self.data_dir,
                                        split="all",
                                        preprocessing=preprocessing,
                                        seed=self.seed)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1 if self.task == "segmentation" else self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
            drop_last=False,
            pin_memory=True)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1 if self.task == "segmentation" else self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
            drop_last=False,
            pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0,
                          shuffle=False,
                          drop_last=False,
                          pin_memory=True)

    def all_dataloader(self):
        return DataLoader(self.all_dataset,
                          batch_size=1,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0,
                          shuffle=False,
                          drop_last=False,
                          pin_memory=True)

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

    @property
    def train_targets(self):
        return self.train_dataset.targets

    @property
    def val_targets(self):
        return self.val_dataset.targets

    @property
    def test_targets(self):
        return self.test_dataset.targets
