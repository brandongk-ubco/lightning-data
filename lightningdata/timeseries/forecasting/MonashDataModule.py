from pytorch_lightning.core.datamodule import LightningDataModule
from ..helpers import get_monash_sets
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY


@DATAMODULE_REGISTRY
class Monash(LightningDataModule):

    category = "timeseries"
    task = "forecasting"

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        assert self.dataset_name in get_monash_sets().keys()