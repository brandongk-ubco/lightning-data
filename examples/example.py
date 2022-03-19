import lightningdata  # noqa: F401
from lightningdata.models import Classifier
import sys
from pytorch_lightning.utilities.cli import LightningCLI


class MyLightningCLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.num_classes",
                              "model.num_classes",
                              apply_on="instantiate")
        parser.link_arguments("data.in_channels",
                              "model.in_channels",
                              apply_on="instantiate")


if __name__ == "__main__":
    cli = MyLightningCLI(Classifier,
                         seed_everything_default=42,
                         trainer_defaults={
                             "gpus": -1,
                             "deterministic": True,
                             "max_epochs": sys.maxsize
                         })
