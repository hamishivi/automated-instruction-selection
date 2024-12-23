"""
Given a list of pickles, aggregate intra-dataset scores and then high-level dataset scores.
Notably, we can aggreagte on inter-dataset level different to the intra-dataset level.
"""
import pickle
import json
from datasets import load_dataset
import argparse
from tqdm import tqdm
import math
from statistics import mean
from transformers import AutoTokenizer
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_files", type=str, nargs="+")  # we can pass in multiple pickles
parser.add_argument("--output_file", type=str)
parser.add_argument("--selection_method", type=str, default="min")  # min, mean, max
parser.add_argument("--output_size", type=int, default=10000)  # number of instances total to select.
parser.add_argument("--train_dataset", type=str)
parser.add_argument("--aggregation_method", type=str, default="minmax")  # mean, minmax, rank
parser.add_argument("--output_dataset", action="store_true")
args = parser.parse_args()

assert args.selection_method in ["min", "max", "mean_min", "mean_max"], "Invalid selection method."
assert args.aggregation_method in ["mean", "minmax"], "Invalid aggregation method."

# load train dataset for printing
if args.train_dataset == "alpaca":
    train_dataset = load_dataset("json", data_files="data/camel_datasets/stanford_alpaca/stanford_alpaca_data.jsonl")[
        "train"
    ]
elif args.train_dataset == "tulu2":
    train_dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
else:
    # assume it's a path to a dataset
    if os.path.exists(args.train_dataset):
        train_dataset = load_dataset("json", data_files=args.train_dataset)["train"]
    else:
        raise ValueError(f"Invalid train dataset {args.train_dataset}.")

tokenizer = AutoTokenizer.from_pretrained("oobabooga/llama-tokenizer")


def compute_influences_for_file(input_file):
    if input_file.endswith(".json"):
        instance_to_influences = json.load(open(input_file, "rb"))
    elif input_file.endswith(".pkl"):
        instance_to_influences = pickle.load(open(input_file, "rb"))

    # two selection methods: min or mean or max
    # this is how we aggregate influences.
    # for mean, we just average scores across all test points.
    # for min, we take the minimum score across all test points. (or max for max.)
    # note for the test points here, top min/max is not the same as when selecting for one dataset.
    # why? because we need to then keep scores around to aggregate. So taking the min across all test points
    # is the closest to what we would get. We could also try

    # map idx -> list of all influences
    all_train_scores = {}
    for i, influences in instance_to_influences.items():
        for train_idx, score in influences.items():
            if train_idx not in all_train_scores:
                all_train_scores[train_idx] = []
            all_train_scores[train_idx].append(score)

    del instance_to_influences

    # track the overall influences/idxes we are taking.
    overall_influences = {}
    if "mean" in args.selection_method:
        print("Using mean influence selection method.")
        # mean reduce
        overall_influences = {k: mean(v) for k, v in all_train_scores.items()}
    elif "min" in args.selection_method:
        print("Using top-min influence selection method.")
        # min reduce
        overall_influences = {k: min(v) for k, v in all_train_scores.items()}
    elif "max" in args.selection_method:
        print("Using top-max influence selection method.")
        # max reduce
        overall_influences = {k: max(v) for k, v in all_train_scores.items()}
    else:
        raise ValueError("Invalid selection method.")
    return overall_influences


# run through all the pickles provided and aggregate the influences.
per_dataset_influences = {}
for input_file in args.input_files:
    print("Processing file: {}".format(input_file))
    per_dataset_influences[input_file] = compute_influences_for_file(input_file)

# inter-dataset aggregation
inter_dataset_scores = {}
if args.aggregation_method == "mean":
    print("Using mean inter-dataset aggregation.")
    collated_influences = {}
    for _, infs in per_dataset_influences.items():
        for idx, score in infs.items():
            if idx not in collated_influences:
                collated_influences[idx] = []
            collated_influences[idx].append(score)
    inter_dataset_scores = {k: mean(v) for k, v in collated_influences.items()}
    fn = sorted if "min" in args.selection_method else lambda x: sorted(x, reverse=True)
    inter_dataset_scores = fn(inter_dataset_scores.items(), key=lambda x: x[1])[: args.output_size]
elif args.aggregation_method == "minmax":
    # base min/max on intra-dataset scores. TODO: more customizability here.
    print("Using minmax inter-dataset aggregation. Using selection method: {}".format(args.selection_method))
    # unlike before, here we round-robin across datasets to avoid score magnitudes affecting this.
    pbar = tqdm(total=args.output_size)
    last_size = 0
    for ds_name, dataset_scores in per_dataset_influences.items():
        per_dataset_influences[ds_name] = sorted(
            dataset_scores.items(), key=lambda x: x[1], reverse="max" in args.selection_method
        )
    while len(inter_dataset_scores) < args.output_size:
        for ds_name in per_dataset_influences.keys():
            if len(inter_dataset_scores) >= args.output_size:
                break
            # sort and pop min/max
            # sorted_scores = sorted(dataset_scores.items(), key=lambda x: x[1], reverse='max' in args.selection_method)
            idx, score = per_dataset_influences[ds_name].pop(0)
            print(f"Dataset: {ds_name}, idx: {idx}, score: {score}")
            # per_dataset_influences[ds_name] = list({k: v for k, v in sorted_scores}.items())
            inter_dataset_scores[idx] = min(score, inter_dataset_scores.get(idx, math.inf))
            pbar.update(len(inter_dataset_scores) - last_size)
            last_size = len(inter_dataset_scores)
    # convert to list
    inter_dataset_scores = list(inter_dataset_scores.items())
else:
    raise ValueError("Invalid aggregation method.")

# construct list of train indices
saved_instances = [int(index) for index, _ in inter_dataset_scores]
saved_scores = [values for _, values in inter_dataset_scores]
assert len(saved_instances) == args.output_size, "Saved instances size does not match output size."

# save the indices
if args.output_dataset:
    output_dataset = []
    for i, train_idx in enumerate(saved_instances):
        if type(train_idx) is not int:
            train_idx = train_idx.item()
        instance = train_dataset[train_idx]
        instance["influence_score"] = saved_scores[i] if type(saved_scores[i]) is float else saved_scores[i].item()
        output_dataset.append(instance)
    with open(args.output_file, "w") as f:
        for instance in output_dataset:
            f.write(json.dumps(instance) + "\n")
else:
    with open(args.output_file, "w") as f:
        json.dump(saved_instances, f, indent=4)
