"""
Given a json with tulu2 data, generate a bar chart showing the distribution of samples.
"""
from matplotlib import pyplot as plt
from datasets import load_dataset
import json
import os
import numpy as np
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_files", nargs="+", type=str)
parser.add_argument("--output_file", type=str)
args = parser.parse_args()

task_names = ["gsm8k", "bbh", "alpacafarm", "tydiqa", "codex", "squad", "mmlu", "full"]


def get_task_name(filename):
    for task_name in task_names:
        if task_name in filename:
            return task_name
    return "other"


# load tulu2
dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
# quick fix: remove the science subsets, collapse into one


def collapse_science(sample):
    if "science" in sample["dataset"]:
        sample["dataset"] = "science"
    return sample


dataset = dataset.map(collapse_science)
counters = []
# compute the counts for each label
for filename in args.input_files:
    # load the json
    selected_samples = json.load(open(filename, "r"))
    # filter the dataset
    subset = dataset.select(selected_samples)
    # count the sources
    counter = Counter(subset["dataset"])
    counters.append(counter)
# add a counter that reflects the overall dataset composition
random_counter = Counter(dataset["dataset"])
# normalize to 10k
normalized_random_counter = {k: v / len(dataset) * 10000 for k, v in random_counter.items()}
counters = [normalized_random_counter] + counters
args.input_files = ["full"] + args.input_files

# go through counters, plot stacked bar chart
fig, ax = plt.subplots(figsize=(10, 5))
# stacked bar chart setup
basenames = [get_task_name(os.path.basename(f)) for f in args.input_files]
all_counter_keys = dataset.unique("dataset")
combined_d = {k: [c[k] for c in counters] for k in all_counter_keys}
width = 0.5
bottom = np.zeros(len(counters))
# colourmap can repeat, stop this
colormap = plt.cm.nipy_spectral
colors = colormap(np.linspace(0, 1, len(all_counter_keys)))
ax.set_prop_cycle("color", colors)
# construct
for ds, ds_count in combined_d.items():
    p = ax.bar(basenames, ds_count, width, label=ds, bottom=bottom)
    bottom += ds_count

ax.set_ylabel("Count")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(args.output_file)
