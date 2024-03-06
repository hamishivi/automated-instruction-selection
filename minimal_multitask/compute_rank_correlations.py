import argparse
import pickle
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import random

parser = argparse.ArgumentParser(description='Compute rank correlations')
parser.add_argument('--pickle_files', type=str, nargs='+', help='List of pickle files')
parser.add_argument('--save_dir', default='results/llama_7b')
parser.add_argument('--file_prefix')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

random_gen = random.Random(args.seed)

datasets = []
for pickle_file in args.pickle_files:
    datasets.append(pickle.load(open(pickle_file, 'rb')))

# compute a sorted list of indices for each index.
dataset_sorted_idxes = {}
dataset_in_order_influences = {}
for dataset, file_name in zip(datasets, args.pickle_files):
    dataset_sorted_idxes[file_name] = []
    dataset_in_order_influences[file_name] = []
    for index in dataset:
        sorted_indices = sorted([i for i in dataset[index]], key=lambda i: dataset[index][i])
        dataset_sorted_idxes[file_name].append(sorted_indices)
        dataset_in_order_influences[file_name].append([dataset[index][i] for i in range(len(dataset[index]))])
    dataset_sorted_idxes[file_name] = np.array(dataset_sorted_idxes[file_name])


print("Inter-dataset rank correlation")
# compute rank correlation between each pair of datasets
# we take this as the average of pairwise correlations between each pair of idxes
for i, file_name1 in enumerate(args.pickle_files):
    for j, file_name2 in enumerate(args.pickle_files):
        correlations = []
        # pick ~100 random pairs of idxes and compute the correlation between them
        pairs1 = np.random.choice(len(dataset_in_order_influences[file_name1]), (100), replace=False)
        pairs2 = np.random.choice(len(dataset_in_order_influences[file_name2]), (100), replace=False)
        pairs = [(i, j) for i, j in zip(pairs1, pairs2)]
        for i, j in pairs:
            correlation, _ = spearmanr(dataset_in_order_influences[file_name1][i], dataset_in_order_influences[file_name2][j])
            correlations.append(correlation)
        print(f'{file_name1} vs {file_name2} Mean:', sum(correlations) / len(correlations))

# now lets do influence correls...
print()
print("Influence rank correlation")
for i, file_name1 in enumerate(args.pickle_files):
    for j, file_name2 in enumerate(args.pickle_files):
        correlations = []
        # pick ~100 random pairs of idxes and compute the correlation between them
        pairs1 = np.random.choice(len(dataset_sorted_idxes[file_name1]), (100), replace=False)
        pairs2 = np.random.choice(len(dataset_sorted_idxes[file_name2]), (100), replace=False)
        pairs = [(i, j) for i, j in zip(pairs1, pairs2)]
        for i, j in pairs:
            correlation, _ = pearsonr(dataset_sorted_idxes[file_name1][i], dataset_sorted_idxes[file_name2][j])
            correlations.append(correlation)
        print(f'{file_name1} vs {file_name2} Mean:', sum(correlations) / len(correlations))

fig, axes = plt.subplots(len(args.pickle_files), len(args.pickle_files), figsize=(20, 20))
# pick a random two indices for each dataset
rands1, rands2 = {}, {}
for filename in args.pickle_files:
    rands1[filename] = int(len(dataset_in_order_influences[filename]) * random_gen.random())
    rands2[filename] = int(len(dataset_in_order_influences[filename]) * random_gen.random())
for i, file_name1 in enumerate(args.pickle_files):
    for j, file_name2 in enumerate(args.pickle_files):
        rand1 = rands1[file_name1]
        rand2 = rands2[file_name2]
        axes[i,j].scatter(x=dataset_in_order_influences[file_name1][rand1], y=dataset_in_order_influences[file_name2][rand2])
        if i == 0:
            axes[i,j].set_title(file_name2.replace("pickles/", "").replace(".pkl", ""))
        if j == 0:
            axes[i,j].set_ylabel(file_name1.replace("pickles/", "").replace(".pkl", ""))
plt.savefig(os.path.join(args.save_dir, f'{args.file_prefix}_influencecorrelation.png'))