'''
Given an influence pickle or json,
compute the quantiles of influence score per test point.
Optionally, also graph a histogram for each test point.
'''
import argparse
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--influence', type=str, required=True)
parser.add_argument('--plot', type=str, default=None)  # if given, plot histogram for each test point
args = parser.parse_args()

print("loading influence scores from", args.influence)
if args.influence.endswith('.json'):
    with open(args.influence, 'r') as f:
        influence = json.load(f)
else:
    with open(args.influence, 'rb') as f:
        influence = pickle.load(f)

num_test_points = len(influence)
if args.plot:
    # setup plot
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle('Influence score distribution per test point')
    figs, axs = plt.subplots(num_test_points // 2 + 1, 2)

print("computing quantiles")
for test_point, influence_scores in influence.items():
    scores = list(influence_scores.values())
    scores.sort()
    scores = np.array(scores)
    print(f'Test point {test_point}:', end='')
    print(f' {scores[0]}', end='')
    print(f' {np.percentile(scores, 25)}', end='')
    print(f' {np.percentile(scores, 50)}', end='')
    print(f' {np.percentile(scores, 75)}', end='')
    print(f' {scores[-1]}')
    if args.plot:
        ax = axs[int(test_point) // 2, int(test_point) % 2]
        ax.hist(scores, bins=100)
        # ax.set_title(f'Test point {test_point}')
        # ax.set_xlabel('Influence score')
        # ax.set_ylabel('Frequency')

if args.plot:
    # for all axes, set x and y limits to be the same
    x_lim_max = max([ax.get_xlim()[1] for ax in axs.flatten()])
    y_lim_max = max([ax.get_ylim()[1] for ax in axs.flatten()])
    x_lim_min = min([ax.get_xlim()[0] for ax in axs.flatten()])
    y_lim_min = min([ax.get_ylim()[0] for ax in axs.flatten()])
    for ax in axs.flatten():
        ax.set_xlim(x_lim_min, x_lim_max)
        ax.set_ylim(y_lim_min, y_lim_max)


    plt.savefig(args.plot)
