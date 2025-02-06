"""
Given a json with tulu2 data, generate a bar chart showing the distribution of samples.
"""
from matplotlib import pyplot as plt
import json
import numpy as np
from collections import Counter, defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_files", nargs="+", type=str)
parser.add_argument("--second_input_files", nargs="+", default=None)
parser.add_argument("--random_sel_file", type=str, default=None)
parser.add_argument("--output_file", type=str)
parser.add_argument("--influence_score_file", type=str, default=None)
parser.add_argument("--normalize_count", action="store_true")
args = parser.parse_args()

counters = []
for file in args.input_files:
    counters.append(Counter())
    with open(file, "r") as f:
        for line in f:
            try:
                counters[-1][json.loads(line)["dataset"]] += 1
            except:
                raise ValueError("Error loading line")

# We assume the second input files have the same number of selection as the first input files
# We will plot two bars for each dataset, one for the first input files and one for the second input files
counters_2 = []
for file in args.second_input_files:
    counters_2.append(Counter())
    with open(file, "r") as f:
        for line in f:
            counters_2[-1][json.loads(line)["dataset"]] += 1

random_counter = None
if args.random_sel_file:
    random_counter = Counter()
    with open(args.random_sel_file, "r") as f:
        for line in f:
            random_counter[json.loads(line)["dataset"]] += 1

evals = ["alpacaeval", "gsm8k", "tydiqa", "bbh", "mmlu", "codex", "squad"]
evals_ordered = sorted(evals)

# The above code plot one par per one dataset. The following code will plot two bar per one dataset and grouped on the x-axis
# go through counters, plot grouped bar chart
fig, ax = plt.subplots(figsize=(10, 5))
# grouped bar chart setup
width = 0.35
basenames = evals_ordered
all_keys = [c.keys() for c in counters]
all_counter_keys = set()
for keys in all_keys:
    all_counter_keys.update(keys)
all_counter_keys = sorted(list(all_counter_keys))
combined_d = {k: [c[k] for c in counters] for k in all_counter_keys}
combined_d_2 = {k: [c[k] for c in counters_2] for k in all_counter_keys}

# Sort the dataset names to ensure consistency
dataset_names = sorted(combined_d.keys())

# Extract the values from both dictionaries in the same order
values_combined_d = np.array([combined_d[name] for name in dataset_names])
values_combined_d_2 = np.array([combined_d_2[name] for name in dataset_names])

# Plotting the stacked bar chart
x = np.arange(len(evals))  # x positions for the bars
width = 0.4  # Width of the bars

fig, ax = plt.subplots(figsize=(12, 6))

# Plot the stacked bars for both dictionaries
bottom_combined_d = np.zeros(len(evals))
bottom_combined_d_2 = np.zeros(len(evals))

colormap = plt.cm.get_cmap('tab20')
def generate_n_colors(cmap, n_colors):
    return [cmap(i / n_colors) for i in range(n_colors)]
colors = generate_n_colors(colormap, len(all_counter_keys))
# colors = colormap(np.linspace(0, 1, len(all_counter_keys)))
for i, (name, color) in enumerate(zip(dataset_names, colors)):
    # Plot for combined_d
    ax.bar(
        x - width / 2, values_combined_d[i], width, bottom=bottom_combined_d, 
        color=color, edgecolor='black', label=name, alpha=0.7
    )
    # Plot for combined_d_2
    ax.bar(
        x + width / 2, values_combined_d_2[i], width, bottom=bottom_combined_d_2, 
        color=color, edgecolor='black', alpha=0.7
    )

    # Update the bottom positions for the next stack
    bottom_combined_d += values_combined_d[i]
    bottom_combined_d_2 += values_combined_d_2[i]

# Set titles and labels
ax.set_xticks(x)
ax.set_xticklabels(evals_ordered)
ax.set_ylabel('Count')
ax.set_title('Stacked Bar Comparison between rds (left) and ccds (right)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

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
