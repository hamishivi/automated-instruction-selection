"""
Given a json with tulu2 data, generate a bar chart showing the distribution of samples.
"""
from matplotlib import pyplot as plt
import json
import numpy as np
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_files", nargs="+", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--influence_score_file", type=str, default=None)
args = parser.parse_args()

counters = []
for file in args.input_files:
    counters.append(Counter())
    with open(file, "r") as f:
        for line in f:
            counters[-1][json.loads(line)["dataset"]] += 1

# go through counters, plot stacked bar chart
fig, ax = plt.subplots(figsize=(10, 5))
# stacked bar chart setup
basenames = ["alpacaeval", "gsm8k", "tydiqa", "bbh", "mmlu", "codex", "squad"]
# [os.path.basename(f).split('_')[0] for f in args.input_files]
all_keys = [c.keys() for c in counters]
all_counter_keys = set()
for keys in all_keys:
    all_counter_keys.update(keys)
all_counter_keys = sorted(list(all_counter_keys))
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

# if we have score annotations, we can also plot those!
if args.influence_score_file:
    fig, ax = plt.subplots(figsize=(10, 5))
    colormap = plt.cm.nipy_spectral
    colors = colormap(np.linspace(0, 1, len(all_counter_keys)))
    ax.set_prop_cycle("color", colors)
    # load influence scores
    dataset_to_scores = {}
    for file in args.input_files:
        with open(file, "r") as f:
            for line in f:
                data = json.loads(line)
                dataset = data["dataset"]
                score = data["influence_score"]
                if dataset not in dataset_to_scores:
                    dataset_to_scores[dataset] = []
                dataset_to_scores[dataset].append(score)
    # plot stacked histogram
    scores = [dataset_to_scores[ds] for ds in all_counter_keys]
    names = all_counter_keys
    ax.hist(scores, bins=20, stacked=True, label=names)
    ax.set_ylabel("Count")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(args.influence_score_file)

# print out the counts
print("Counts:")
for k, c in counters[0].items():
    print(f"{k}: {c}")
