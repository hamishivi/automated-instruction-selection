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

parser = argparse.ArgumentParser()
parser.add_argument("--input_files", nargs="+", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--selection_method", type=str, default="min")  # min, mean, max
parser.add_argument("--output_size", type=int, default=10000)  # number of instances total to select.
parser.add_argument("--train_datasets", nargs="+", type=str, default=["alpaca"])  # alpaca, tulu2
# save the output dataset in text. must be used for mult-file inputs.
parser.add_argument("--output_dataset", action="store_true", default=False)
parser.add_argument("--domain_weights", type=str)  # json file containing domain weights normalized to 1.
parser.add_argument("--select_only_from_file", type=str)  # only select instances from this file.
args = parser.parse_args()

assert args.selection_method in ["min", "max", "mean_min", "mean_max", "normalized_mean_min", "normalized_mean_max"], "Invalid selection method."
# assert len(args.input_files) == len(args.train_datasets), "Number of input files must match number of train datasets, " + len(args.input_files) + ", " + len(args.train_datasets)
if len(args.input_files) > 1:
    assert args.output_dataset, "Must save output dataset for multiple input files."
assert args.output_file, "Must specify output file."

# load instance info
instance_to_influences_list = []
for input_file in args.input_files:
    instance_to_influences_list.append(pickle.load(open(input_file, "rb")))

if args.select_only_from_file:
    with open(args.select_only_from_file) as f:
        subsample_ids = set([json.loads(line)["id"] for line in f])

# load domain weight information
if args.domain_weights:
    domain_weights = json.load(open(args.domain_weights))
    # normalize domain weights just in case
    domain_weights = {k: v / sum(domain_weights.values()) for k, v in domain_weights.items()}
    # domain max size
    domain_max_size = {k: v * args.output_size for k, v in domain_weights.items()}
else:
    domain_max_size = None


def get_domain_values(domain):
    if "science" in domain and "science" in domain_max_size:
        return domain_max_size["science"]
    elif domain not in domain_max_size:
        return 0
    return domain_max_size[domain]


def remove_dupes_ordered(lst):
    seen = set()
    new_lst = []
    for item in lst:
        if item not in seen:
            new_lst.append(item)
            seen.add(item)
    return new_lst


# load train datasets for printing
train_datasets = []
for train_dataset in args.train_datasets:
    if train_dataset == "alpaca":
        train_datasets.append(
            load_dataset("json", data_files="data/camel_datasets/stanford_alpaca/stanford_alpaca_data.jsonl")["train"]
        )
    elif train_dataset == "tulu2":
        train_datasets.append(load_dataset("allenai/tulu-v2-sft-mixture", split="train"))
    else:
        # assume it's a path to a dataset
        if os.path.exists(train_dataset):
            train_datasets.append(load_dataset("json", data_files=train_dataset)["train"])
        else:
            raise ValueError(f"Invalid train dataset {train_dataset}.")
# just assume llama tokenizer for now. This is used for debugging mainly.
tokenizer = AutoTokenizer.from_pretrained("oobabooga/llama-tokenizer")

# flatten instance to influence. now the train_idx has format (train_dataset_idx, train_idx)
# this lets us treat the train_idx as a unique identifier.
instance_to_influences = {}
for i, instance_to_influences_i in enumerate(instance_to_influences_list):
    for test_index, influences in enumerate(instance_to_influences_i):
        # sometimes i saved a dict of test idx -> influences, instead of a list.
        # in this case, just grab the interior dict.
        if type(influences) is int:
            assert influences == test_index, "Test index unexpected."
            influences = instance_to_influences_i[influences]
        if test_index not in instance_to_influences:
            instance_to_influences[test_index] = {}
        for train_idx, score in influences.items():
            if train_idx == -1:
                # malformed influence score, skip it.
                continue
            instance_to_influences[test_index][(i, train_idx)] = score

# track domain sizes
domain_sizes = defaultdict(int)

# two selection methods: min or mean or max
# for mean_min, we simply compute the average influence score for each train point, and then select the top k.
# for mean_max, we have the inverted form of the above.
# for min, we select the top k points per test instance such that we have a total of k instances.
# max is simply min but reversed, for cosine sim or toxigen inversion.
if "normalized_mean" in args.selection_method:
    print("Using normalized mean influence selection method.")
    # this is mean, but we normalize influence scores to be in [-1, 1] range.
    average_influences = {}
    for i, influences in instance_to_influences.items():
        for train_idx, score in influences.items():
            if train_idx not in average_influences:
                average_influences[train_idx] = []
            average_influences[train_idx].append(score)
    average_influences = list(average_influences.items())
    # normalize scores. notably, we normalize per test instance, not per train instance
    for idx in range(len(average_influences[0][1])):
        scores_for_test_point = [scores[idx] for _, scores in average_influences]
        min_scores = min(scores_for_test_point)
        max_scores = max(scores_for_test_point)
        # normalize scores
        scores_for_test_point = [(score - min_scores) / (max_scores - min_scores) for score in scores_for_test_point]
        for i, (train_idx, scores) in enumerate(average_influences):
            average_influences[i][1][idx] = scores_for_test_point[i]
    # then average
    average_influences = [(train_idx, mean(scores)) for train_idx, scores in average_influences]
    # sort by average influence
    if "min" in args.selection_method:
        average_influences = sorted(average_influences, key=lambda x: x[1])[: args.output_size]
    else:
        average_influences = sorted(average_influences, key=lambda x: x[1])[-args.output_size:]
    # construct list of train indices
    saved_instances = [index for index, _ in average_influences]
    saved_scores = [score for _, score in average_influences]
elif "mean" in args.selection_method:
    print("Using mean influence selection method.")
    average_influences = {}
    for i, influences in instance_to_influences.items():
        for train_idx, score in influences.items():
            if train_idx not in average_influences:
                average_influences[train_idx] = []
            average_influences[train_idx].append(score)
    average_influences = list(average_influences.items())
    average_influences = [(train_idx, mean(scores)) for train_idx, scores in average_influences]
    # sort by average influence
    if "min" in args.selection_method:
        average_influences = sorted(average_influences, key=lambda x: x[1])[: args.output_size]
    else:
        average_influences = sorted(average_influences, key=lambda x: x[1])[-args.output_size:]
    # construct list of train indices
    saved_instances = [index for index, _ in average_influences]
    saved_scores = [score for _, score in average_influences]
elif "min" in args.selection_method:
    print("Using top-min influence selection method.")
    # round-robin the instances, taking until we hit the output size.
    saved_instances = []
    saved_scores = []
    last_size = 0
    sorted_instance_to_influence = {}
    for test_d, influences in instance_to_influences.items():
        sorted_influences = sorted(influences.items(), key=lambda x: x[1])
        sorted_instance_to_influence[test_d] = sorted_influences
    with tqdm(total=args.output_size) as pbar:
        while len(saved_instances) < args.output_size:
            for test_d, influences in instance_to_influences.items():
                # pop off the smallest influence
                inst, score = sorted_instance_to_influence[test_d].pop(0)
                # if we have a -inf or inf score, skip it.
                if score == float("-inf") or score == float("inf"):
                    continue
                # if we have set domain weights, make sure we don't exceed the max size.
                # will only work with tulu data.
                inst_0 = inst[0] if type(inst[0]) is int else inst[0].item()
                inst_1 = inst[1] if type(inst[1]) is int else inst[1].item()
                domain = train_datasets[inst_0][inst_1]["dataset"]
                inst_text = train_datasets[inst_0][inst_1]["messages"][0]["content"].strip()
                if "science" in domain:
                    domain = "science"
                if domain_max_size:
                    # pop until we get a domain we can use, or we run out of data.
                    while domain_sizes[domain] >= get_domain_values(domain):
                        if len(sorted_instance_to_influence[test_d]) == 0:
                            raise ValueError("Not enough instances to satisfy domain size constraints.")
                        inst, score = sorted_instance_to_influence[test_d].pop(0)
                        inst_0 = inst[0] if type(inst[0]) is int else inst[0].item()
                        inst_1 = inst[1] if type(inst[1]) is int else inst[1].item()
                        domain = train_datasets[inst_0][inst_1]["dataset"]
                        inst_text = train_datasets[inst_0][inst_1]["messages"][0]["content"].strip()
                        if "science" in domain:
                            domain = "science"
                    domain_sizes[domain] += 1

                # if we are only selecting from a file, make sure we only select from that file.
                if args.select_only_from_file:
                    sample_id = train_datasets[inst_0][inst_1]["id"]
                    if sample_id not in subsample_ids:
                        continue

                saved_instances.append(inst)
                saved_scores.append(score)
                # set list the saved instances in case of dups.
                prev_size = len(saved_instances)
                saved_instances = remove_dupes_ordered(saved_instances)
                # if it was a dup, remove the score.
                # also remove the domain increment.
                if len(saved_instances) < prev_size:
                    saved_scores = saved_scores[:-1]
                    domain_sizes[domain] -= 1
                # update pbar
                pbar.update(len(saved_instances) - last_size)
                last_size = len(saved_instances)
                # if we have hit max size, break out
                if len(saved_instances) >= args.output_size:
                    break
                # print(domain, domain_sizes[domain], get_domain_values(domain), domain_sizes, len(saved_instances), sum(domain_sizes.values()))
    # if we are over the output size, remove the last few instances.
    saved_instances = saved_instances[: args.output_size]
elif "max" in args.selection_method:
    print("Using top-max influence selection method.")
    # round-robin the instances, taking until we hit the output size.
    saved_instances = []
    saved_scores = []
    last_size = 0
    sorted_instance_to_influence = {}
    for test_d, influences in instance_to_influences.items():
        sorted_influences = sorted(influences.items(), key=lambda x: x[1], reverse=True)
        sorted_instance_to_influence[test_d] = sorted_influences
    with tqdm(total=args.output_size) as pbar:
        while len(saved_instances) < args.output_size:
            for test_d, influences in sorted_instance_to_influence.items():
                # pop off the largest influence
                inst, score = sorted_instance_to_influence[test_d].pop(0)

                inst_0 = inst[0] if type(inst[0]) is int else inst[0].item()
                inst_1 = inst[1] if type(inst[1]) is int else inst[1].item()
                if args.select_only_from_file:
                    sample_id = train_datasets[inst_0][inst_1]["id"]
                    if sample_id not in subsample_ids:
                        continue
                
                saved_instances.append(inst)
                saved_scores.append(score)
                # set list the saved instances in case of dups.
                prev_size = len(saved_instances)
                saved_instances = remove_dupes_ordered(saved_instances)
                # if it was a dup, remove the score.
                if len(saved_instances) < prev_size:
                    saved_scores = saved_scores[:-1]
                # update pbar
                pbar.update(len(saved_instances) - last_size)
                last_size = len(saved_instances)
    # if we are over the output size, remove the last few instances.
    saved_instances = saved_instances[: args.output_size]

saved_instances = remove_dupes_ordered(saved_instances)
print(f"Saving {len(saved_instances)} instances")
# if we are outputting the actual dataset, time to save
# add the influence score to the instance for plotting and save
if args.output_dataset:
    output_dataset = []
    for i, (dataset_idx, train_idx) in enumerate(saved_instances):
        if type(dataset_idx) is not int:
            dataset_idx = dataset_idx.item()
        if type(train_idx) is not int:
            train_idx = train_idx.item()
        instance = train_datasets[dataset_idx][train_idx]
        instance["influence_score"] = saved_scores[i] if type(saved_scores[i]) is float else saved_scores[i].item()
        output_dataset.append(instance)
    with open(args.output_file, "w") as f:
        for instance in output_dataset:
            f.write(json.dumps(instance) + "\n")
else:
    assert len(train_datasets) == 1, "Can only output dataset idxes with single input file."
    # convert indices to actual ints and drop the dataset index, since we only have one.
    saved_instances = [int(i[1]) for i in saved_instances]
    # save top instances
    if not args.output_file:
        args.output_file = f"{args.input_file.split('.')[0]}_{args.selection_method}{args.output_size}.json"
    with open(args.output_file, "w") as f:
        json.dump(saved_instances, f, indent=4)
