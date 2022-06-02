from .CompressedNpzDataset import CompressedNpzDataset
from ..VisionDataModule import VisionDataModule
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY


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

    Dataset = CityScapesDataset
