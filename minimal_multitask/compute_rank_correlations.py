import argparse
import pickle
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='Compute rank correlations')
parser.add_argument('--pickle_files', type=str, nargs='+', help='List of pickle files')
args = parser.parse_args()

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
        # pick ~10 random pairs of idxes and compute the correlation between them
        pairs1 = np.random.choice(len(dataset_sorted_idxes[file_name1]), (100), replace=False)
        pairs2 = np.random.choice(len(dataset_sorted_idxes[file_name2]), (100), replace=False)
        pairs = [(i, j) for i, j in zip(pairs1, pairs2)]
        for i, j in pairs:
            correlation, _ = spearmanr(dataset_sorted_idxes[file_name1][i], dataset_sorted_idxes[file_name2][j])
            correlations.append(correlation)
        print(f'{file_name1} vs {file_name2} Mean:', sum(correlations) / len(correlations))

print()
print("top-100 rank correlation")
# top-100 rank correls instead. We just take the top-100 from the first dataset and compare to order in the second dataset
for i, file_name1 in enumerate(args.pickle_files):
    for j, file_name2 in enumerate(args.pickle_files):
        correlations = []
        # pick ~10 random pairs of idxes and compute the correlation between them
        pairs1 = np.random.choice(len(dataset_sorted_idxes[file_name1]), (100), replace=False)
        pairs2 = np.random.choice(len(dataset_sorted_idxes[file_name2]), (100), replace=False)
        pairs = [(i, j) for i, j in zip(pairs1, pairs2)]
        for i, j in pairs:
            top_100 = dataset_sorted_idxes[file_name1][i][:100]
            ordered_second = [x for x in dataset_sorted_idxes[file_name2][j] if x in top_100]
            correlation, _ = spearmanr(top_100, ordered_second)
            correlations.append(correlation)
        print(f'{file_name1} vs {file_name2} Mean:', sum(correlations) / len(correlations))

# now lets do influence correls...
print()
print("Influence rank correlation")
for i, file_name1 in enumerate(args.pickle_files):
    for j, file_name2 in enumerate(args.pickle_files):
        correlations = []
        # pick ~10 random pairs of idxes and compute the correlation between them
        pairs1 = np.random.choice(len(dataset_sorted_idxes[file_name1]), (100), replace=False)
        pairs2 = np.random.choice(len(dataset_sorted_idxes[file_name2]), (100), replace=False)
        pairs = [(i, j) for i, j in zip(pairs1, pairs2)]
        for i, j in pairs:
            correlation, _ = pearsonr(dataset_sorted_idxes[file_name1][i], dataset_sorted_idxes[file_name2][j])
            correlations.append(correlation)
        print(f'{file_name1} vs {file_name2} Mean:', sum(correlations) / len(correlations))

# plot a set of scatter subplots
import matplotlib.pyplot as plt
import random

random_gen = random.Random(42)
fig, axes = plt.subplots(len(args.pickle_files), len(args.pickle_files), figsize=(20, 20))
for i, file_name1 in enumerate(args.pickle_files):
    for j, file_name2 in enumerate(args.pickle_files):
        rand1 = int(len(dataset_in_order_influences[file_name1]) * random_gen.random())
        rand2 = int(len(dataset_in_order_influences[file_name2]) * random_gen.random())
        axes[i,j].scatter(x=dataset_in_order_influences[file_name1][rand1], y=dataset_in_order_influences[file_name2][rand2])
        if i == 0:
            axes[i,j].set_title(file_name2.replace("pickles/", "").replace(".pkl", ""))
        if j == 0:
            axes[i,j].set_ylabel(file_name1.replace("pickles/", "").replace(".pkl", ""))
plt.savefig('plots.png')