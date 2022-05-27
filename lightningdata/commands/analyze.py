from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from ._analyze_segmentation import analyze_segmentation
from ._analyze_classification import analyze_classification


def analyze(data: str, overwrite=False, augment_policy_path: str = None):
    Dataset = DATAMODULE_REGISTRY[data]
    dataset = Dataset(num_workers=0,
                      batch_size=1,
                      augment_policy_path=augment_policy_path)
    if dataset.category == "vision":
        if dataset.task == "classification":
            analyze_classification(dataset, overwrite)
        if dataset.task == "segmentation":
            analyze_segmentation(dataset, overwrite)
