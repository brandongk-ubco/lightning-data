from .CompressedNpzDataset import CompressedNpzDataset
from ..VisionDataModule import VisionDataModule
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY


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

    Dataset = SeverstalDataset
