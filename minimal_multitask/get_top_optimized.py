import os
import pickle
import json
from datasets import load_dataset
import argparse
from tqdm import tqdm
from statistics import mean
from transformers import AutoTokenizer
from collections import defaultdict
import heapq

parser = argparse.ArgumentParser()
parser.add_argument("--input_files", nargs="+", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--selection_method", type=str, default="max")  # Now default is "max"
parser.add_argument("--output_size", type=int, default=10000)  # number of instances total to select.
parser.add_argument("--train_datasets", nargs="+", type=str, default=["alpaca"])  # alpaca, tulu2
# Save the output dataset in text. Must be used for multi-file inputs.
parser.add_argument("--output_dataset", action="store_true", default=False)
parser.add_argument("--domain_weights", type=str)  # JSON file containing domain weights normalized to 1.
parser.add_argument("--select_only_from_file", type=str)  # Only select instances from this file.
args = parser.parse_args()

assert args.selection_method in [
    "min",
    "max",
    "mean_min",
    "mean_max",
    "normalized_mean_min",
    "normalized_mean_max",
], "Invalid selection method."
if len(args.input_files) > 1:
    assert args.output_dataset, "Must save output dataset for multiple input files."
assert args.output_file, "Must specify output file."

# Load instance info
instance_to_influences_list = []
for input_file in args.input_files:
    if input_file.endswith(".json"):
        with open(input_file, 'r') as f:
            instance_to_influences_list.append(json.load(f))
    elif input_file.endswith(".pkl"):
        with open(input_file, "rb") as f:
            instance_to_influences_list.append(pickle.load(f))
    else:
        raise ValueError(f"Invalid input file {input_file}.")

if args.select_only_from_file:
    with open(args.select_only_from_file) as f:
        subsample_ids = set(json.loads(line)["id"] for line in f)

# Load domain weight information
if args.domain_weights:
    with open(args.domain_weights) as f:
        domain_weights = json.load(f)
    # Normalize domain weights
    total_weight = sum(domain_weights.values())
    domain_weights = {k: v / total_weight for k, v in domain_weights.items()}
    # Domain max size
    domain_max_size = {k: v * args.output_size for k, v in domain_weights.items()}
else:
    domain_max_size = None

def get_domain_max_size(domain):
    if "science" in domain and "science" in domain_max_size:
        return domain_max_size["science"]
    return domain_max_size.get(domain, 0)

# Load train datasets
train_datasets = []
for train_dataset in args.train_datasets:
    if train_dataset == "alpaca":
        train_datasets.append(
            load_dataset(
                "json", data_files="data/camel_datasets/stanford_alpaca/stanford_alpaca_data.jsonl"
            )["train"]
        )
    elif train_dataset == "tulu2":
        train_datasets.append(load_dataset("allenai/tulu-v2-sft-mixture", split="train"))
    else:
        if os.path.exists(train_dataset):
            train_datasets.append(load_dataset("json", data_files=train_dataset)["train"])
        else:
            raise ValueError(f"Invalid train dataset {train_dataset}.")

# Flatten instance to influence
instance_to_influences = {}
for i, instance_to_influences_i in enumerate(instance_to_influences_list):
    for test_index, influences in enumerate(instance_to_influences_i):
        if isinstance(influences, (int, str)):
            influences = instance_to_influences_i[influences]
        if test_index not in instance_to_influences:
            instance_to_influences[test_index] = {}
        for train_idx, score in influences.items():
            if train_idx == -1:
                continue  # Skip malformed influence score
            instance_to_influences[test_index][(i, train_idx)] = score

# Track domain sizes
domain_sizes = defaultdict(int)

# Initialize seen_instances set to keep track of duplicates
seen_instances = set()

# Implement the "max" selection method
if "max" in args.selection_method:
    print("Using top-max influence selection method.")
    saved_instances = []
    saved_scores = []
    heap = []

    # Build a max-heap by storing negative scores
    for test_d, influences in instance_to_influences.items():
        for train_idx, score in influences.items():
            # Use negative score for max-heap
            heapq.heappush(heap, (-score, train_idx))

    with tqdm(total=args.output_size) as pbar:
        while len(saved_instances) < args.output_size and heap:
            neg_score, inst = heapq.heappop(heap)
            score = -neg_score  # Convert back to positive score

            if inst in seen_instances:
                continue  # Skip duplicates

            dataset_idx, train_idx = inst
            dataset_idx = int(dataset_idx)
            train_idx = int(train_idx)

            instance = train_datasets[dataset_idx][train_idx]

            if args.select_only_from_file:
                sample_id = instance["id"]
                if sample_id not in subsample_ids:
                    continue

            # Domain handling
            domain = instance.get("dataset", "")
            if "science" in domain:
                domain = "science"

            if domain_max_size and domain_sizes[domain] >= get_domain_max_size(domain):
                continue  # Skip if domain max size reached

            saved_instances.append(inst)
            saved_scores.append(score)
            seen_instances.add(inst)
            domain_sizes[domain] += 1

            pbar.update(1)

    # Ensure we have exactly the desired output size
    saved_instances = saved_instances[: args.output_size]
    saved_scores = saved_scores[: args.output_size]

print(f"Saving {len(saved_instances)} instances")
# Output dataset
if args.output_dataset:
    output_dataset = []
    for i, (dataset_idx, train_idx) in enumerate(saved_instances):
        dataset_idx = int(dataset_idx)
        train_idx = int(train_idx)
        instance = train_datasets[dataset_idx][train_idx]
        instance["influence_score"] = float(saved_scores[i])
        output_dataset.append(instance)
    with open(args.output_file, "w") as f:
        for instance in output_dataset:
            f.write(json.dumps(instance) + "\n")
else:
    assert (
        len(train_datasets) == 1
    ), "Can only output dataset indices with a single input file."
    # Convert indices to actual ints and drop the dataset index
    saved_indices = [int(i[1]) for i in saved_instances]
    if not args.output_file:
        args.output_file = f"{args.input_files[0].split('.')[0]}_{args.selection_method}{args.output_size}.json"
    with open(args.output_file, "w") as f:
        json.dump(saved_indices, f, indent=4)
