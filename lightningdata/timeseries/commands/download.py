from lightningdata.helpers import get_data_dir
import os
import json
import sys
import requests
from zipfile import ZipFile
import shutil
from tqdm import tqdm
from ..helpers import get_monash_sets


def download(name: str = None):

    if name is None:
        print("Pick from one of:")
        print("\n\t".join(monash_sets.keys()))
        sys.exit()

    data_dir = get_data_dir(name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    monash_set = monash_sets[name]

    zip_path = os.path.join(data_dir, f"{name}.zip")
    print("Downloading")
    with requests.get(monash_set, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            pbar = tqdm(total=int(r.headers['Content-Length']))
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
            shutil.copyfileobj(r.raw, f)
    print("")
    print("Extracting")
    with ZipFile(zip_path, 'r') as zip:
        zip.extractall(data_dir)
