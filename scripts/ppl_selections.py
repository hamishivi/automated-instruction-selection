import argparse
from typing import List
from datasets import load_dataset
import os
import json
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--ppl_scores", type=str, nargs="+", default=["selections/ppl_200k_exp/nlls.pkl"],
                    help="List of paths to perplexity score files.")
parser.add_argument("--train_datasets", type=str, nargs="+", default=["/net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/200k_exps/unbalanced_distribution_200k.jsonl"],
                    help="List of paths to training datasets.")
parser.add_argument("--output_size", type=int, default=10000,
                    help="Number of instances total to select.")
parser.add_argument("--output_file_path", type=str, required=True,
                    help="Path to save the selected instances.")
parser.add_argument("--mid_ppl", action="store_true",
                    help="Flag to enable mid perplexity filtering.")
args = parser.parse_args()

train_datasets = []
for train_dataset in args.train_datasets:
    if train_dataset == "alpaca":
        train_dataset = load_dataset("json", data_files="data/camel_datasets/stanford_alpaca/stanford_alpaca_data.jsonl")["train"]
    elif train_dataset == "tulu2":
        train_dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
    else:
        if os.path.exists(train_dataset):
            train_dataset = load_dataset("json", data_files=train_dataset)["train"]
        else:
            raise ValueError(f"Invalid train dataset {train_dataset}.")
    train_datasets.append(train_dataset)



all_nlls = []
for idx, file in enumerate(args.ppl_scores):
    if not os.path.exists(file):
        raise ValueError(f"Invalid ppl scores file {file}.")
    with open(file, "rb") as f:
        nlls = pickle.load(f)
    # key in nlls is data idx, replace with (file idx, data idx)
    new_nlls = {}
    for key, value in nlls.items():
        new_nlls[(idx, key)] = value
    all_nlls.append(new_nlls)

# flatten all_nlls
nlls = {}
for nll in all_nlls:
    nlls.update(nll)

# Sort by perplexity score, from highest to lowest
sorted_nlls = sorted(nlls.items(), key=lambda x: x[1], reverse=True)
if not args.mid_ppl:
    selected_indices = [x[0] for x in sorted_nlls[:args.output_size]]
else:
    # selecting from the 33% - 66% percentile
    selected_indices = [x[0] for x in sorted_nlls[int(len(sorted_nlls) * 0.33): int(len(sorted_nlls) * 0.33) + args.output_size]]

output_dataset = []
for idx, (file_idx, data_idx) in enumerate(selected_indices):
    instance = train_datasets[file_idx][data_idx]
    instance["influence_score"] = sorted_nlls[idx][1]
    output_dataset.append(instance)
with open(args.output_file_path, "w") as f:
    for instance in output_dataset:
        f.write(json.dumps(instance) + "\n")
