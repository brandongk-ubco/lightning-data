from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from ._analyze_segmentation import analyze_segmentation
from ._analyze_classification import analyze_classification


def analyze(data: str, overwrite=False, augment_policy_path: str = None):
    Dataset = DATAMODULE_REGISTRY[data]
    if Dataset.task == "classification":
        analyze_classification(data, overwrite, augment_policy_path)
    if Dataset.task == "segmentation":
        analyze_segmentation(data, overwrite, augment_policy_path)
