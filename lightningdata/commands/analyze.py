from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from tqdm import tqdm
import torch
import pandas as pd
import os
from ..helpers import composite_SD
import pprint
import json
from scipy.ndimage.measurements import label
import numpy as np


def collect_stats(dataset):
    dataloader = dataset.all_dataloader()
    dataiter = iter(dataloader)
    rows = []

    structure = np.ones((3, 3), dtype=np.int)
    for i, sample in tqdm(enumerate(dataiter), total=len(dataiter)):
        x, y = sample
        name = dataset.all_data[i]

        row = {
            "name": name,
            "width": x.shape[-1],
            "height": x.shape[-2],
            "channels": x.shape[-3]
        }
        for j in range(row["channels"]):
            row[f"channel_{j}_mean"] = float(x[:, j, :, :].mean())
            row[f"channel_{j}_std"] = float(x[:, j, :, :].std())

        for j, clazz in enumerate(dataset.classes):
            clazz_sample = y[0, j, :, :]
            labeled, ncomponents = label(clazz_sample, structure)
            row[f"class_{clazz}_coverage"] = float(clazz_sample.sum() /
                                                   torch.numel(clazz_sample))
            row[f"class_{clazz}_components"] = ncomponents
        rows.append(row)
    df = pd.DataFrame(rows)

    return df


def analyze(data: str, overwrite=False, augment_policy_path: str = None):

    Dataset = DATAMODULE_REGISTRY[data]

    dataset = Dataset(num_workers=0, batch_size=1, augment_policy_path=augment_policy_path)

    stats_file = os.path.join(dataset.data_dir, "stats.csv")
    if not os.path.exists(stats_file) or overwrite:
        stats_df = collect_stats(dataset)
        stats_df.to_csv(stats_file, index=False)
    else:
        stats_df = pd.read_csv(stats_file)

    channels = [
        int(n[8:-5])
        for n in stats_df.columns
        if n.startswith("channel_") and n.endswith("_mean")
    ]

    channel_stats = {}

    for channel in channels:
        channel_stats[channel] = {
            "mean":
                stats_df[f"channel_{channel}_mean"].mean(),
            "std":
                composite_SD(stats_df[f"channel_{channel}_mean"],
                             stats_df[f"channel_{channel}_std"],
                             len(stats_df[f"channel_{channel}_mean"])),
        }

    class_stats = {}

    for clazz in dataset.classes:
        clazz_df = stats_df[stats_df[f"class_{clazz}_coverage"] > 0].copy(
            deep=True).reset_index(drop=True)
        class_stats[clazz] = {
            "mean_coverage": clazz_df[f"class_{clazz}_coverage"].mean(),
            "mean_components": clazz_df[f"class_{clazz}_components"].mean(),
            "count": len(clazz_df)
        }

    stats = {"channels": channel_stats, "classes": class_stats}

    pprint.pprint(stats, indent=4)
    with open(os.path.join(dataset.data_dir, "stats_aggregate.json"),
              "w") as jsonfile:
        json.dump(stats, jsonfile, indent=4)
