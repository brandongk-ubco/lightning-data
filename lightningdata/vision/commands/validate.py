from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from tqdm import tqdm


def validate(data: str, augment_policy_path: str = None):

    Dataset = DATAMODULE_REGISTRY[data]

    dataset = Dataset(augment_policy_path=augment_policy_path,
                      num_workers=0,
                      batch_size=1)

    dataloader = dataset.train_dataloader()
    for dataloader in [
            dataset.train_dataloader(),
            dataset.val_dataloader(),
            dataset.test_dataloader()
    ]:
        dataiter = iter(dataloader)
        expected_x_shape = None
        for sample in tqdm(dataiter):
            x, y = sample
            if expected_x_shape is None:
                expected_x_shape = x.shape
            assert x.shape == expected_x_shape
            assert x.dim() == 4
            assert x.shape[0] == 1
            assert x.shape[1] == dataset.in_channels
            assert y.shape[0] == 1
            assert y.shape[1] == dataset.num_classes
            assert x[0, 0, :, :].min() >= 0.
            assert x[0, 0, :, :].max() <= 1.
            if dataset.task == "classification":
                assert y.dim() == 2
            elif dataset.task == "segmentation":
                assert y.dim() == 4
                assert x.shape[-2:] == y.shape[-2:]
            else:
                raise ValueError(
                    "Expected dataset task to be either classification or segmentation"
                )
