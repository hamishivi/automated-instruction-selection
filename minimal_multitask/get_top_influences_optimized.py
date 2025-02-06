"""
Given an influence pickle, select the top k instances based on the selection method.
Works with multiple pickles, where you pass the given train dataset with each pickle.
This means you can compute influences over sharded datasets and then combine them.
"""
import os
import pickle
import json
from datasets import load_dataset
import argparse
from tqdm import tqdm
from statistics import mean
from transformers import AutoTokenizer
from collections import defaultdict
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input_files", nargs="+", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--selection_method", type=str, default="min")  # min, mean, max
parser.add_argument("--output_size", type=int, default=10000)  # number of instances total to select.
parser.add_argument("--train_datasets", nargs="+", type=str, default=["alpaca"])  # alpaca, tulu2
parser.add_argument("--output_dataset", action="store_true", default=False)
parser.add_argument("--domain_weights", type=str)  # json file containing domain weights normalized to 1.
parser.add_argument("--select_only_from_file", type=str)  # only select instances from this file.
args = parser.parse_args()

assert args.selection_method in ["min", "max", "mean_min", "mean_max", "normalized_mean_min", "normalized_mean_max"], "Invalid selection method."
if len(args.input_files) > 1:
    assert args.output_dataset, "Must save output dataset for multiple input files."
assert args.output_file, "Must specify output file."

def load_influence_data(file_path):
    if file_path.endswith(".pkl"):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        with open(file_path, "r") as f:
            return json.load(f)

# Load instance info
instance_to_influences_list = [load_influence_data(file) for file in args.input_files]

if args.select_only_from_file:
    with open(args.select_only_from_file) as f:
        subsample_ids = set(json.loads(line)["id"] for line in f)

# Load domain weight information
if args.domain_weights:
    with open(args.domain_weights) as f:
        domain_weights = json.load(f)
    total_weight = sum(domain_weights.values())
    domain_max_size = {k: (v / total_weight) * args.output_size for k, v in domain_weights.items()}
else:
    domain_max_size = None

def get_domain_values(domain):
    if domain_max_size is None:
        return float('inf')
    if "science" in domain and "science" in domain_max_size:
        return domain_max_size["science"]
    return domain_max_size.get(domain, 0)

# Load train datasets for printing
def load_train_dataset(dataset_name):
    if dataset_name == "alpaca":
        return load_dataset("json", data_files="data/camel_datasets/stanford_alpaca/stanford_alpaca_data.jsonl")["train"]
    elif dataset_name == "tulu2":
        return load_dataset("allenai/tulu-v2-sft-mixture", split="train")
    elif os.path.exists(dataset_name):
        return load_dataset("json", data_files=dataset_name)["train"]
    else:
        raise ValueError(f"Invalid train dataset {dataset_name}.")

train_datasets = [load_train_dataset(dataset) for dataset in args.train_datasets]

tokenizer = AutoTokenizer.from_pretrained("oobabooga/llama-tokenizer")

# Flatten instance to influence
instance_to_influences = defaultdict(dict)
for i, instance_to_influences_i in enumerate(instance_to_influences_list):
    for test_index, influences in enumerate(instance_to_influences_i):
        if isinstance(influences, (int, str)):
            influences = instance_to_influences_i[influences]
        for train_idx, score in influences.items():
            if train_idx != -1:
                instance_to_influences[test_index][(i, train_idx)] = score

# Track domain sizes
domain_sizes = defaultdict(int)

def process_normalized_mean():
    average_influences = defaultdict(list)
    for influences in instance_to_influences.values():
        for train_idx, score in influences.items():
            average_influences[train_idx].append(score)
    
    average_influences = list(average_influences.items())
    scores_array = np.array([scores for _, scores in average_influences])
    
    min_scores = np.min(scores_array, axis=0)
    max_scores = np.max(scores_array, axis=0)
    normalized_scores = (scores_array - min_scores) / (max_scores - min_scores)
    
    average_influences = [(train_idx, np.mean(normalized_scores[i])) for i, (train_idx, _) in enumerate(average_influences)]
    
    if "min" in args.selection_method:
        average_influences.sort(key=lambda x: x[1])
    else:
        average_influences.sort(key=lambda x: -x[1])
    
    return [index for index, _ in average_influences[:args.output_size]], [score for _, score in average_influences[:args.output_size]]

def process_mean():
    average_influences = defaultdict(list)
    for influences in instance_to_influences.values():
        for train_idx, score in influences.items():
            average_influences[train_idx].append(score)
    
    average_influences = [(train_idx, np.mean(scores)) for train_idx, scores in average_influences.items()]
    
    if "min" in args.selection_method:
        average_influences.sort(key=lambda x: x[1])
    else:
        average_influences.sort(key=lambda x: -x[1])
    
    return [index for index, _ in average_influences[:args.output_size]], [score for _, score in average_influences[:args.output_size]]

def process_min_max():
    saved_instances = []
    saved_scores = []
    sorted_instance_to_influence = {test_d: sorted(influences.items(), key=lambda x: x[1], reverse=("max" in args.selection_method))
                                    for test_d, influences in instance_to_influences.items()}
    
    with tqdm(total=args.output_size) as pbar:
        while len(saved_instances) < args.output_size:
            for test_d, influences in sorted_instance_to_influence.items():
                if not influences:
                    continue
                inst, score = influences.pop(0)
                if score in (float("-inf"), float("inf")):
                    continue
                
                inst_0, inst_1 = map(lambda x: x if isinstance(x, int) else x.item(), inst)
                domain = train_datasets[inst_0][inst_1]["dataset"]
                if "science" in domain:
                    domain = "science"
                
                if domain_max_size:
                    while domain_sizes[domain] >= get_domain_values(domain):
                        if not influences:
                            break
                        inst, score = influences.pop(0)
                        inst_0, inst_1 = map(lambda x: x if isinstance(x, int) else x.item(), inst)
                        domain = train_datasets[inst_0][inst_1]["dataset"]
                        if "science" in domain:
                            domain = "science"
                    if domain_sizes[domain] >= get_domain_values(domain):
                        continue
                    domain_sizes[domain] += 1
                
                if args.select_only_from_file and train_datasets[inst_0][inst_1]["id"] not in subsample_ids:
                    continue
                
                saved_instances.append(inst)
                saved_scores.append(score)
                pbar.update(1)
                
                if len(saved_instances) >= args.output_size:
                    break
    
    return saved_instances[:args.output_size], saved_scores[:args.output_size]

if "normalized_mean" in args.selection_method:
    print("Using normalized mean influence selection method.")
    saved_instances, saved_scores = process_normalized_mean()
elif "mean" in args.selection_method:
    print("Using mean influence selection method.")
    saved_instances, saved_scores = process_mean()
elif "min" in args.selection_method or "max" in args.selection_method:
    print(f"Using top-{'min' if 'min' in args.selection_method else 'max'} influence selection method.")
    saved_instances, saved_scores = process_min_max()

saved_instances = list(set(saved_instances))
print(f"Saving {len(saved_instances)} instances")

if args.output_dataset:
    with open(args.output_file, "w") as f:
        for i, (dataset_idx, train_idx) in enumerate(saved_instances):
            dataset_idx = dataset_idx if isinstance(dataset_idx, int) else dataset_idx.item()
            train_idx = int(train_idx) if isinstance(train_idx, str) else (train_idx if isinstance(train_idx, int) else train_idx.item())
            instance = train_datasets[dataset_idx][train_idx]
            instance["influence_score"] = saved_scores[i] if isinstance(saved_scores[i], float) else saved_scores[i].item()
            f.write(json.dumps(instance) + "\n")
else:
    assert len(train_datasets) == 1, "Can only output dataset idxes with single input file."
    saved_instances = [int(i[1]) for i in saved_instances]
    with open(args.output_file, "w") as f:
        json.dump(saved_instances, f, indent=4)
