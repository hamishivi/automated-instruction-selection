"""
Delete all my datasets.
"""
from datetime import datetime

import pytz
from beaker import Beaker
from tqdm import tqdm

utc = pytz.UTC
beak = Beaker.from_env(default_workspace="ai2/hamishi")

for dataset in tqdm(beak.workspace.iter_datasets(workspace="ai2/hamishi")):
    if dataset.created < utc.localize(datetime(2023, 1, 1)):
        # print(f"Deleting {dataset.name}... {dataset}")
        beak.dataset.delete(dataset)
        # print(dataset.name, dataset.created)
print("done!")
