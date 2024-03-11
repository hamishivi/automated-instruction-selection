'''
Given an influence pickle, select the top k instances based on the selection method.
Only works with one pickle.
'''
import pickle
import json
from datasets import load_dataset
import argparse
from tqdm import tqdm
from statistics import mean
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--selection_method', type=str, default='min') # min, mean, max
parser.add_argument('--output_size', type=int, default=10000) # number of instances total to select.
parser.add_argument('--train_dataset', type=str, default='alpaca')
args = parser.parse_args()

assert args.selection_method in ['min', 'max', 'mean_min', 'mean_max'], "Invalid selection method."
assert args.train_dataset in ['alpaca', 'tulu2'], "Invalid train dataset."

# instance info
instance_to_influences = pickle.load(open(args.input_file, "rb"))

# load train dataset for printing
if args.train_dataset == 'alpaca':
    train_dataset = load_dataset('json', data_files='data/camel_datasets/stanford_alpaca/stanford_alpaca_data.jsonl')['train']
else:
    train_dataset = load_dataset('allenai/tulu-v2-sft-mixture', split='train')

tokenizer = AutoTokenizer.from_pretrained('oobabooga/llama-tokenizer')

# two selection methods: min or mean or max
# for mean_min, we simply compute the average influence score for each train point, and then select the top k.
# for mean_max, we have the inverted form of the above.
# for min, we select the top k points per test instance such that we have a total of k instances.
# max is simply min but reversed, for cosine sim or toxigen inversion.
if 'mean' in args.selection_method:
    print("Using mean influence selection method.")
    average_influences = {}
    for i, influences in instance_to_influences.items():
        for train_idx, score in influences.items():
            if train_idx not in average_influences:
                average_influences[train_idx] = []
            average_influences[train_idx].append(score)
    average_influences = list(average_influences.items())
    # sort by average influence
    if 'min' in args.selection_method:
        average_influences = sorted(average_influences, key=lambda x: mean(x[1]))[:args.output_size]
    else:
        average_influences = sorted(average_influences, key=lambda x: mean(x[1]))[-args.output_size:]
    # construct list of train indices
    saved_instances = [index for index, _ in average_influences]
elif 'min' in args.selection_method:
    print("Using top-min influence selection method.")
    # round-robin the instances, taking until we hit the output size.
    saved_instances = []
    last_size = 0
    with tqdm(total=args.output_size) as pbar:
        while len(saved_instances) < args.output_size:
            for test_d, influences in instance_to_influences.items():
                sorted_influences = sorted(influences.items(), key=lambda x: x[1])
                # pop off the smallest influence
                saved_instances.append(sorted_influences.pop(0)[0])
                # update the influences
                instance_to_influences[test_d] = {k: v for k, v in sorted_influences}
                # set list the saved instances in case of dups.
                saved_instances = list(set(saved_instances))
                # update pbar
                pbar.update(len(saved_instances) - last_size)
                last_size = len(saved_instances)
    # if we are over the output size, remove the last few instances.
    saved_instances = saved_instances[:args.output_size]
elif 'max' in args.selection_method:
    print("Using top-max influence selection method.")
    # round-robin the instances, taking until we hit the output size.
    saved_instances = []
    last_size = 0
    with tqdm(total=args.output_size) as pbar:
        while len(saved_instances) < args.output_size:
            for test_d, influences in instance_to_influences.items():
                sorted_influences = sorted(influences.items(), key=lambda x: x[1], reverse=True)
                # pop off the largest influence
                saved_instances.append(sorted_influences.pop(0)[0])
                # update the influences
                instance_to_influences[test_d] = {k: v for k, v in sorted_influences}
                # set list the saved instances in case of dups.
                saved_instances = list(set(saved_instances))
                # update pbar
                pbar.update(len(saved_instances) - last_size)
                last_size = len(saved_instances)
    # if we are over the output size, remove the last few instances.
    saved_instances = saved_instances[:args.output_size]

saved_instances = list(set(saved_instances))
# convert indices to actual ints.
saved_instances = [int(i) for i in saved_instances]

print(f"Saved {len(saved_instances)} instances")
# save top instances
with open(args.output_file, "w") as f:
    json.dump(saved_instances, f, indent=4)
