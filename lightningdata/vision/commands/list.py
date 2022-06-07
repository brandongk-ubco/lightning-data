import pprint
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY


def list():
    pprint.pprint(DATAMODULE_REGISTRY)
