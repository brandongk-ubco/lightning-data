from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from matplotlib import pyplot as plt
import torch
from argh import arg
import pytorch_lightning
import cv2


@arg("--split", choices=["train", "val", "test", "all"])
def visualize(data: str,
              split: str = "train",
              augment_policy_path: str = None,
              rows=None,
              columns=None,
              min_img_width=128,
              min_img_height=128,
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
    max_width = 1920
    max_height = 1080
    img, _ = next(dataiter)

    img_width = max(img.shape[-1], min_img_width)
    img_height = max(img.shape[-1], min_img_height)

    if rows is None:
        rows = max(max_height // img_height, 1)

    if columns is None:
        columns = max(max_width // img_width, 1)

    img_shape = (img_width, img_height)

    shown = 0
    num_figs = rows * columns
    for i, sample in enumerate(dataiter):
        if shown >= examples:
            break
        idx = i % num_figs

        if idx == 0:
            plt.close()
            fig = plt.figure(figsize=(11, 11))

        x, y = sample

        x = torch.squeeze(x)

        fig.add_subplot(rows, columns, idx + 1)

        if x.dim() not in [2, 3]:
            raise ValueError(
                f"Expected either 2 (grayscale) or 3 (RGB) channel images, found {x.dim()}"
            )

        if x.dim() == 3:
            x = x.moveaxis(0, -1)

        cmap = "gray" if x.dim() == 2 else None

        x = x.numpy()

        if img.shape != img_shape:
            x = cv2.resize(x, img_shape)

        plt.imshow(x, cmap=cmap)

        if dataset.task == "classification":
            plt.title(dataset.classes[y.argmax()], fontdict={'fontsize': 8})
        plt.axis('off')

        if idx == num_figs - 1:
            shown += 1

            # TODO: This needs to be dymaically calculated as it's based on image size...
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.4,
                                hspace=0.4)
            plt.show()
