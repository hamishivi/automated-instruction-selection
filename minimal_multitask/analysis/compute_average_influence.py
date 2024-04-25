import pickle
import json
import argparse
from statistics import mean, stdev

parser = argparse.ArgumentParser()
parser.add_argument('--influence_files', type=str, nargs="+")
parser.add_argument('--selection_result', type=str)
parser.add_argument('--aggregation_method', default='mean')

args = parser.parse_args()

if args.selection_result is not None:
    with open(args.selection_result, 'r') as f:
        selected_index = json.load(f)

for input_file in args.influence_files:
    print("Processing file: {}".format(input_file))
    instance_to_influences = pickle.load(open(input_file, "rb"))

    all_train_scores = {}
    for i, influences in instance_to_influences.items():
        for train_idx, score in influences.items():
            if train_idx not in all_train_scores:
                all_train_scores[train_idx] = []
            all_train_scores[train_idx].append(score)
    
    overall_influences = {}
    if 'mean' in args.aggregation_method:
        print("Using mean influence selection method.")
        # mean reduce
        overall_influences = {k: mean(v) for k, v in all_train_scores.items()}
    elif 'min' in args.aggregation_method:
        print("Using top-min influence selection method.")
        # min reduce
        overall_influences = {k: min(v) for k, v in all_train_scores.items()}
    elif 'max' in args.aggregation_method:
        print("Using top-max influence selection method.")
        # max reduce
        overall_influences = {k: max(v) for k, v in all_train_scores.items()}
    else:
        raise ValueError("Invalid selection method.")
    
    mean_scores = []
    if args.selection_result is not None:
        for index in selected_index:
            mean_scores.append(overall_influences[index])
    else:
        for index in overall_influences:
            mean_scores.append(overall_influences[index]) 
    
    print("Influence score stat for {}, Mean: {}, Std: {}".format(input_file, mean(mean_scores), stdev(mean_scores)))