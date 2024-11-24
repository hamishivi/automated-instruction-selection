import argparse
from datasets import load_dataset
import os
import json
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--ppl_scores", type=str, default="selections/ppl_200k_exp/nlls.pkl")
parser.add_argument("--train_dataset", type=str, default="/net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/200k_exps/unbalanced_distribution_200k.jsonl")
parser.add_argument("--output_size", type=int, default=10000)  # number of instances total to select.)
parser.add_argument("--output_file_path", type=str)
parser.add_argument("--mid_ppl", action="store_true")
args = parser.parse_args()


if args.train_dataset == "alpaca":
    train_dataset = load_dataset("json", data_files="data/camel_datasets/stanford_alpaca/stanford_alpaca_data.jsonl")["train"]
elif args.train_dataset == "tulu2":
    train_dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
else:
    if os.path.exists(args.train_dataset):
        train_dataset = load_dataset("json", data_files=args.train_dataset)["train"]
    else:
        raise ValueError(f"Invalid train dataset {train_dataset}.")

with open(args.ppl_scores, "rb") as f:
    nlls = pickle.load(f)

# Sort by perplexity score, from highest to lowest
sorted_nlls = sorted(nlls.items(), key=lambda x: x[1], reverse=True)
if not args.mid_ppl:
    selected_indices = [x[0] for x in sorted_nlls[:args.output_size]]
else:
    # selecting from the 33% - 66% percentile
    selected_indices = [x[0] for x in sorted_nlls[int(len(sorted_nlls)*0.33) : int(len(sorted_nlls)*0.33)+args.output_size]]

output_dataset = []
for idx, data_idx in enumerate(selected_indices):
    instance = train_dataset[data_idx]
    instance["influence_score"] = sorted_nlls[idx][1]
    output_dataset.append(instance)
with open(args.output_file_path, "w") as f:
    for instance in output_dataset:
        f.write(json.dumps(instance) + "\n")
