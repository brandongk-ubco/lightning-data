from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from matplotlib import pyplot as plt
import torch
from argh import arg
import pytorch_lightning


@arg("--split", choices=["train", "val", "test", "all"])
def visualize(data: str,
              split: str = "train",
              augment_policy_path: str = None,
              rows=None,
              columns=None,
              examples=1,
              seed=42):

    pytorch_lightning.seed_everything(seed)

    Dataset = DATAMODULE_REGISTRY[data]

    dataset = Dataset(augment_policy_path=augment_policy_path,
                      num_workers=0,
                      batch_size=1)

    if split == "train":
        dataloader = dataset.train_dataloader()
    elif split == "val":
        dataloader = dataset.val_dataloader()
    elif split == "test":
        dataloader = dataset.test_dataloader()
    elif split == "all":
        dataloader = dataset.all_dataloader()

    dataiter = iter(dataloader)

    if rows is None or columns is None:
        img, _ = next(dataiter)
        max_width = 1920
        max_height = 1080
        columns = max(max_width // img.shape[-1], 1)
        rows = max(max_height // img.shape[-2], 1)

    shown = 0
    num_figs = rows * columns
    for i, sample in enumerate(dataiter):
        if shown >= examples:
            break
        idx = i % num_figs

        if idx == 0:
            plt.close()
            fig = plt.figure(figsize=(11, 11))

        x, _ = sample

        x = torch.squeeze(x)

        fig.add_subplot(rows, columns, idx + 1)

        if x.dim() == 2:
            plt.imshow(x, cmap="gray")
        elif x.dim() == 3:
            plt.imshow(x.moveaxis(0, -1))
        else:
            raise ValueError(
                f"Expected either 2 (grayscale) or 3 (RGB) channel images, found {x.dim()}"
            )
        plt.axis('off')

        if idx == num_figs - 1:
            shown += 1
            plt.show()
