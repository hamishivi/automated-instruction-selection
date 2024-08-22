import argparse
from matplotlib import pyplot as plt
import pickle
import os
import json
import random
from tqdm import tqdm

random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--input_files', type=str, nargs='+')  # we can pass in multiple pickles!
parser.add_argument('--output_file', type=str) # where to save the graph to. uses save_fig
parser.add_argument('--selection_method', type=str, default='min') # min, mean, max, random
parser.add_argument('--color_idxes', type=str) # json file with idxes of datapoints to color.
args = parser.parse_args()

assert args.selection_method in ['min', 'max', 'mean', 'random'], "Invalid selection method."

# load the pickles
influences = [pickle.load(open(f, "rb")) for f in args.input_files]

# color idxes
if args.color_idxes:
    color_idxes = json.load(open(args.color_idxes, "rb"))

# compute aggregate score per-dataset based on selection method.
def compute_dataset_influences(influence_dict):
    overall_influences = {}
    for _, influence_scores in influence_dict.items():
        for index, score in influence_scores.items():
            if index not in overall_influences:
                overall_influences[index] = []
            overall_influences[index].append(score)
    if args.selection_method == 'mean':
        return {k: sum(v) / len(v) for k, v in overall_influences.items()}
    elif args.selection_method == 'min':
        return {k: min(v) for k, v in overall_influences.items()}
    elif args.selection_method == 'max':
        return {k: max(v) for k, v in overall_influences.items()}
    elif args.selection_method == 'random':
        return {k: random.choice(v) for k, v in overall_influences.items()}
    
scores = [compute_dataset_influences(influence) for influence in tqdm(influences)]    
import pdb; pdb.set_trace()
# now, pairwise scatter plots.
fig, axs = plt.subplots(len(scores), len(scores), figsize=(25, 25))
for i in range(len(scores)):
    labels = ['blue' if k not in color_idxes else 'red' for k in scores[i].keys()]
    for j in range(len(scores)):        
        if i == 0:
            axs[i, j].set_title(os.path.basename(args.input_files[j]).replace(".pkl", ""))
        if j == 0:
            axs[i, j].set_ylabel(os.path.basename(args.input_files[i]).replace(".pkl", ""))
        if i == j:
            axs[i, j].hist(list(scores[i].values()), bins=20)
        else:
            axs[i, j].scatter(list(scores[i].values()), list(scores[j].values()), c=labels, cmap='coolwarm')
            # plot lines at 0 to show quadrants.
            axs[i, j].axhline(y=0, color='r', linestyle='--')
            axs[i, j].axvline(x=0, color='r', linestyle='--')

# save the figure
plt.tight_layout()
plt.savefig(args.output_file)
