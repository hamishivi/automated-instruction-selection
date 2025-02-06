from datasets import load_dataset

import argparse 
import json
import random

parser = argparse.ArgumentParser()
parser.add_argument("--ori_dataset", type=str, default="allenai/tulu-v2-sft-mixture")
parser.add_argument("--distribution_file", type=str, default="auxiliary_data/tulu2_distrib.json")
parser.add_argument("--subset_size", type=int, default=10000)
parser.add_argument("--output_file", type=str, default="data/customized_datasets/tulu2_10k.jsonl")
parser.add_argument("--seed", type=int, default=42)

random.seed(42)

args = parser.parse_args()

if args.ori_dataset.endswith(".json") or args.ori_dataset.endswith(".jsonl"):
    ds = load_dataset("json", data_files=args.ori_dataset)["train"]
else:
    ds = load_dataset(args.ori_dataset, split="train")
with open(args.distribution_file, 'r') as f:
    source_share = json.load(f)

samples_per_source = {source: int(args.subset_size * share) for source, share in source_share.items()}

# Store sampled data
sampled_data = []

# Perform sampling for each source
for source, num_samples in samples_per_source.items():
    # Filter rows belonging to the current source
    source_rows = ds.filter(lambda row: row["dataset"] == source)

    # Randomly sample the required number of rows from the source
    sampled_rows = source_rows.shuffle(seed=42).select(range(num_samples))

    # Extend the sampled data list
    sampled_data.extend(sampled_rows)

# Shuffle the combined sampled data to ensure randomness
random.shuffle(sampled_data)

# Convert sampled data to list of dicts (for saving to JSON)
sampled_data_dicts = [dict(row) for row in sampled_data]

with open(args.output_file, 'w') as f:
    for row in sampled_data_dicts:
        f.write(json.dumps(row) + '\n')