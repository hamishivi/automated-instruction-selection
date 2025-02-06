from collections import defaultdict
import argparse
import json
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name_or_path", type=str, default="allenai/tulu-v2-sft-mixture")
parser.add_argument("--output_file", type=str, default="auxiliary_data/tulu2_distrib.json")

args = parser.parse_args()

if args.dataset_name_or_path.endswith(".json") or args.dataset_name_or_path.endswith(".jsonl"):
    dataset = load_dataset("json", data_files=args.dataset_name_or_path)["train"]
else:
    dataset = load_dataset(args.dataset_name_or_path, split="train")
distrib = defaultdict(int)
for example in dataset:
    distrib[example["dataset"]] += 1

for source, count in distrib.items():
    distrib[source] = count / len(dataset)

with open(args.output_file, "w") as f:
    json.dump(distrib, f)