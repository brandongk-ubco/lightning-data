from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from argh import arg
from matplotlib import pyplot as plt
import torch


def visualize(data: str, augment_policy_path: str = None, rows=10, columns=10):

    Dataset = DATAMODULE_REGISTRY[data]

    dataset = Dataset(augment_policy_path=augment_policy_path,
                      num_workers=0,
                      batch_size=1)
    dataloader = dataset.train_dataloader()
    dataiter = iter(dataloader)

    num_figs = rows * columns
    for i, sample in enumerate(dataiter):
        idx = i % num_figs

        if idx == 0:
            plt.close()
            fig = plt.figure(figsize=(11, 11))

        x, y = sample
        clazz_idx = int(torch.argmax(y))
        clazz = dataset.classes[clazz_idx]
        fig.add_subplot(rows, columns, idx + 1)
        img = torch.squeeze(x[0, :, :, :])
        if img.dim() == 2:
            plt.imshow(img, cmap="gray")
        elif img.dim() == 3:
            plt.imshow(img.moveaxis(0, -1))
        else:
            raise ValueError(
                f"Expected either 2 (grayscale) or 3 (RGB) channel images, found {img.dim()}"
            )
        plt.axis('off')

        if idx == num_figs - 1:
            plt.show()
