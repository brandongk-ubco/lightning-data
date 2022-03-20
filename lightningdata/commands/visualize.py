from turtle import xcor
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from matplotlib import pyplot as plt
import torch


def visualize(data: str,
              augment_policy_path: str = None,
              rows=None,
              columns=None,
              examples=1):

    Dataset = DATAMODULE_REGISTRY[data]

    dataset = Dataset(augment_policy_path=augment_policy_path,
                      num_workers=0,
                      batch_size=1)
    dataloader = dataset.train_dataloader()
    dataiter = iter(dataloader)

    if rows is None or columns is None:
        img, _ = next(dataiter)
        max_width = 1920
        max_height = 1080
        columns = max_width // img.shape[-2]
        rows = max_height // img.shape[-1]

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
