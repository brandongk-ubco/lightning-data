from tqdm import tqdm
import pandas as pd
import os
from ..helpers import composite_SD
import pprint
import json


def collect_stats(dataset):
    dataloader = dataset.all_dataloader()
    dataiter = iter(dataloader)
    rows = []

    for i, sample in tqdm(enumerate(dataiter), total=len(dataiter)):
        x, y = sample

        row = {
            "name": i,
            "width": x.shape[-1],
            "height": x.shape[-2],
            "channels": x.shape[-3]
        }
        for j in range(row["channels"]):
            row[f"channel_{j}_mean"] = float(x[:, j, :, :].mean())
            row[f"channel_{j}_std"] = float(x[:, j, :, :].std())
            row["class"] = dataset.classes[y.argmax()]
        rows.append(row)
    df = pd.DataFrame(rows)

    return df


def analyze_classification(dataset, overwrite=False):

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

    class_stats = stats_df["class"].value_counts().to_dict()

    stats = {"channels": channel_stats, "classes": class_stats}

    pprint.pprint(stats, indent=4)
    with open(os.path.join(dataset.data_dir, "stats_aggregate.json"),
              "w") as jsonfile:
        json.dump(stats, jsonfile, indent=4)
