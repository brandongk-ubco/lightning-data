from lightningdata import Datasets
from argh import arg
from matplotlib import pyplot as plt
import torch


@arg('dataset', choices=Datasets.choices())
def visualize(dataset: Datasets,
              augment_policy_path: str = None,
              rows=10,
              columns=10):

    Dataset = Datasets.get(dataset)
    dataset = Dataset(augment_policy_path=augment_policy_path,
                      num_workers=0,
                      batch_size=1)
    dataloader = dataset.train_dataloader()
    dataiter = iter(dataloader)

    fig = plt.figure(figsize=(11, 11))
    num_figs = rows * columns
    for i in range(1, num_figs + 1):
        x, y = next(dataiter)
        clazz_idx = int(torch.argmax(y))
        clazz = dataset.classes[clazz_idx]
        fig.add_subplot(rows, columns, i)
        img = torch.squeeze(x[0, :, :, :])
        plt.imshow(img, cmap="gray")
        plt.axis('off')

    plt.show()
