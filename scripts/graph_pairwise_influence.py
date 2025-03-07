import argparse
from matplotlib import pyplot as plt
import pickle
import os
import random
import json
from tqdm import tqdm

random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--input_files", type=str, nargs="+")  # we can pass in multiple pickles!
parser.add_argument("--output_file", type=str)  # where to save the graph to. uses save_fig
parser.add_argument("--selection_method", type=str, default="min")  # min, mean, max, random
parser.add_argument("--color_idxes", type=str, nargs="+")  # json file with idxes of datapoints to color.
parser.add_argument("--colour_top_k", type=int)  # colour the higher k values in the scatter plot.
parser.add_argument("--subsample", type=int)  # subsample the data to make the scatter plot more readable and faster.
args = parser.parse_args()

assert args.selection_method in ["min", "max", "mean", "random"], "Invalid selection method."


def open_file(filename):
    if filename.endswith(".json"):
        return json.load(open(filename))
    elif filename.endswith(".pkl"):
        return pickle.load(open(filename, "rb"))


# load the pickles
influences = [open_file(f) for f in tqdm(args.input_files)]

# color idxes
if args.color_idxes:
    color_idxes = []
    for idx, color_idx in enumerate(args.color_idxes):
        color_idxes.append([int(line) for line in open(color_idx)])
else:
    color_idxes = []

# compute aggregate score per-dataset based on selection method.


def compute_dataset_influences(influence_dict):
    overall_influences = {}
    for _, influence_scores in influence_dict.items():
        for index, score in influence_scores.items():
            if index not in overall_influences:
                overall_influences[index] = []
            overall_influences[index].append(score)
    if args.selection_method == "mean":
        return {k: sum(v) / len(v) for k, v in overall_influences.items()}
    elif args.selection_method == "min":
        return {k: min(v) for k, v in overall_influences.items()}
    elif args.selection_method == "max":
        return {k: max(v) for k, v in overall_influences.items()}
    elif args.selection_method == "random":
        return {k: random.choice(v) for k, v in overall_influences.items()}


scores = [compute_dataset_influences(influence) for influence in tqdm(influences)]
# subsample if needed. Prefer to subsample from color_idxes.
new_scores = []
if args.subsample:
    print(f"Subsampling to {args.subsample} datapoints.")
    index_range = range(len(scores[0]))
    random_idxes = random.sample(index_range, args.subsample)
    for i in tqdm(range(len(scores))):
        new_scores.append({})
        for idx in random_idxes:
            if idx in scores[i]:
                new_scores[i][idx] = scores[i][idx]
            elif str(idx) in scores[i]:
                new_scores[i][idx] = scores[i][str(idx)]
            else:
                print(f"Index {idx} not found in dataset {i}.")
                raise ValueError
scores = new_scores

# now, pairwise scatter plots.
fig, axs = plt.subplots(len(scores), len(scores), figsize=(25, 25))
for i in range(len(scores)):
    for j in range(len(scores)):
        if i == 0:
            axs[i, j].set_title(os.path.basename(args.input_files[j]).replace(".pkl", ""))
        if j == 0:
            axs[i, j].set_ylabel(os.path.basename(args.input_files[i]).replace(".pkl", ""))
        if i == j:
            axs[i, j].hist(list(scores[i].values()), bins=20)
        else:
            labels = []
            for k in scores[i].keys():
                if k in color_idxes[i]:
                    labels.append("red")
                elif k in color_idxes[j]:
                    labels.append("green")
                else:
                    labels.append("blue")
            axs[i, j].scatter(list(scores[i].values()), list(scores[j].values()), c=labels, cmap="coolwarm")
            # plot lines at 0 to show quadrants.
            axs[i, j].axhline(y=0, color="r", linestyle="--")
            axs[i, j].axvline(x=0, color="r", linestyle="--")

# save the figure
plt.tight_layout()
plt.savefig(args.output_file)
