import lightningdata
import sys
from pytorch_lightning.utilities.cli import LightningCLI
from Model import Classifier

if __name__ == "__main__":

    mnist = lightningdata.Datasets.get("mnist")

    cli = LightningCLI(Classifier,
                       mnist,
                       seed_everything_default=42,
                       trainer_defaults={
                           "gpus": -1,
                           "deterministic": True,
                           "max_epochs": sys.maxsize
                       })
