import argparse
import pickle
from scipy.stats import spearmanr, pearsonr
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import json

parser = argparse.ArgumentParser(description='Compute rank correlations')
parser.add_argument('--pickle_files', type=str, nargs='+', help='List of pickle files')
parser.add_argument('--save_dir', default='results/llama_7b')
parser.add_argument('--file_prefix')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_samples', type=int, default=100)
parser.add_argument('--num_points', type=int, default=-1)
parser.add_argument('--fix_indices', action='store_true')
parser.add_argument('--minus_one', help='some datasets have an extra -1 key', action='store_true')
parser.add_argument('--highlight_selection', help='the corresponding selection for each ', nargs='+')
args = parser.parse_args()

random_gen = random.Random(args.seed)

datasets = []
for pickle_file in args.pickle_files:
    datasets.append(pickle.load(open(pickle_file, 'rb')))

selections = []
if args.highlight_selection is not None:
    for selection_fp in args.highlight_selection:
        with open(selection_fp, 'r') as f:
            selection = json.load(f)
        selections.append(selection)

# compute a sorted list of indices for each index.
dataset_sorted_idxes = {}
dataset_in_order_influences = {}
for dataset, file_name in zip(datasets, args.pickle_files):
    dataset_sorted_idxes[file_name] = []
    dataset_in_order_influences[file_name] = []
    for index in dataset:
        sorted_indices = sorted([i for i in dataset[index]], key=lambda i: dataset[index][i])
        dataset_sorted_idxes[file_name].append(sorted_indices)
        if args.minus_one:
            # temporary -1 for alpacaeval multiround selection, since there is an extra -1
            dataset_in_order_influences[file_name].append(np.array([dataset[index][i] for i in range(len(dataset[index]) - 1)]))
        else:
            dataset_in_order_influences[file_name].append(np.array([dataset[index][i] for i in range(len(dataset[index]))]))
        # Filter out anomaly
        dataset_in_order_influences[file_name][-1][dataset_in_order_influences[file_name][-1] > 100] = 0
        dataset_in_order_influences[file_name][-1][dataset_in_order_influences[file_name][-1] < -100] = 0
    dataset_sorted_idxes[file_name] = np.array(dataset_sorted_idxes[file_name])

print("Computing influence score shift by initial influence score")
initial_inf_score = []
inf_shift = []
for i, file_name1 in enumerate(args.pickle_files):
    for j, file_name2 in enumerate(args.pickle_files[i:]):
        if file_name1 == file_name2:
            continue
        correlations = []
        # pick ~100 random pairs of idxes and compute the correlation between them
        pairs = [(i, i) for i in range(args.num_samples)]
        for i, j in pairs:
            difference = dataset_in_order_influences[file_name2][i] - dataset_in_order_influences[file_name1][j]
            initial_inf_score += list(dataset_in_order_influences[file_name1][j])
            inf_shift += list(difference)

        pearson, _ = pearsonr(initial_inf_score, inf_shift)
        spearman, _ = spearmanr(initial_inf_score, inf_shift)

        if args.num_points > -1:
            initial_inf_score_sampled, inf_shift_sampled = zip(*random.sample(list(zip(initial_inf_score, inf_shift)), k=args.num_points))
        plt.scatter(x=initial_inf_score_sampled, y=inf_shift_sampled, color='green')
        plt.xlabel("Initial Influence Score")
        plt.ylabel("Influence Score Shift")
        plt.suptitle(f"Pearson: {pearson:.3f}, Spearman: {spearman:.3f}")
        plt.savefig(os.path.join(args.save_dir, f'{args.file_prefix}_influence_score_shift.png'))
        plt.close()

print("Computing influence score shift by initial influence score, averaged")
for i, file_name1 in enumerate(args.pickle_files):
    for j, file_name2 in enumerate(args.pickle_files[i:]):
        if file_name1 == file_name2:
            continue
        initial_inf_score = None
        inf_shift = None
        # pick ~100 random pairs of idxes and compute the correlation between them
        pairs = [(i, i) for i in range(args.num_samples)]
        for i, j in pairs:
            if inf_shift is None:
                inf_shift = dataset_in_order_influences[file_name2][i] - dataset_in_order_influences[file_name1][j]
                initial_inf_score = dataset_in_order_influences[file_name1][j]
            else:
                inf_shift += dataset_in_order_influences[file_name2][i] - dataset_in_order_influences[file_name1][j]
                initial_inf_score += dataset_in_order_influences[file_name1][j]
        initial_inf_score /= args.num_samples
        inf_shift /= args.num_samples

        pearson, _ = pearsonr(initial_inf_score, inf_shift)
        spearman, _ = spearmanr(initial_inf_score, inf_shift)
        plt.scatter(x=initial_inf_score, y=inf_shift, color='green')
        plt.xlabel("Initial Influence Score")
        plt.ylabel("Influence Score Shift")
        plt.suptitle(f"Pearson: {pearson:.3f}, Spearman: {spearman:.3f}")
        plt.savefig(os.path.join(args.save_dir, f'{args.file_prefix}_influence_score_shift_avg.png'))
        plt.scatter(x=initial_inf_score[selections[0] + selections[1]], y=inf_shift[selections[0] + selections[1]], color='blue')
        plt.savefig(os.path.join(args.save_dir, f'{args.file_prefix}_influence_score_shift_avg_highlighted.png'))
        plt.close()

print()
print("Computing influence score shift by rank")
for i, file_name1 in enumerate(args.pickle_files):
    for j, file_name2 in enumerate(args.pickle_files[i:]):
        if file_name1 == file_name2:
            continue
        correlations = []
        # pick ~100 random pairs of idxes and compute the correlation between them
        pairs = [(i, i) for i in range(args.num_samples)]
        inf_shift_rank = []
        for i, j in pairs:
            cur_difference = []
            for idx in dataset_sorted_idxes[file_name1][i]:
                difference = datasets[1][i][idx] - datasets[0][i][idx]
                if abs(difference) > 100:
                    print(f"Anomoly influence difference at idx: {idx}, inf difference {difference:.3f}")
                    continue
                cur_difference.append(difference)
            inf_shift_rank.append(cur_difference)
        inf_shift_rank = np.array(inf_shift_rank)
        inf_shift_rank_avg = np.average(inf_shift_rank, axis=0)

        plt.scatter(x=range(len(inf_shift_rank_avg)), y=inf_shift_rank_avg, alpha=0.01)
        plt.xlabel("Initial Influence rank")
        plt.ylabel("Influence Score Shift")
        pearson, _ = pearsonr(range(len(inf_shift_rank_avg)), inf_shift_rank_avg)
        spearman, _ = spearmanr(range(len(inf_shift_rank_avg)), inf_shift_rank_avg)
        plt.suptitle(f"Pearson: {pearson:.3f}, Spearman: {spearman:.3f}")
        plt.savefig(os.path.join(args.save_dir, f'{args.file_prefix}_influence_score_shift_by_rank.png'))
        plt.close()
