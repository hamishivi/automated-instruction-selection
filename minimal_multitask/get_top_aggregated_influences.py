'''
Given a list of pickles, aggregate intra-dataset scores and then high-level dataset scores.
Notably, we can aggreagte on inter-dataset level different to the intra-dataset level.
'''
import pickle
import json
from datasets import load_dataset
import argparse
from tqdm import tqdm
from statistics import mean
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--input_files', type=str, nargs='+')  # we can pass in multiple pickles
parser.add_argument('--output_file', type=str)
parser.add_argument('--selection_method', type=str, default='min') # min, mean, max
parser.add_argument('--output_size', type=int, default=10000) # number of instances total to select.
parser.add_argument('--train_dataset', type=str, default='alpaca')
parser.add_argument('--aggregation_method', type=str, default='minmax') # mean, minmax
args = parser.parse_args()

assert args.selection_method in ['min', 'max', 'mean_min', 'mean_max'], "Invalid selection method."
assert args.train_dataset in ['alpaca', 'tulu2'], "Invalid train dataset."
assert args.aggregation_method in ['mean', 'minmax'], "Invalid aggregation method."

# load train dataset for printing
if args.train_dataset == 'alpaca':
    train_dataset = load_dataset('json', data_files='data/camel_datasets/stanford_alpaca/stanford_alpaca_data.jsonl')['train']
else:
    train_dataset = load_dataset('allenai/tulu-v2-sft-mixture', split='train')

tokenizer = AutoTokenizer.from_pretrained('oobabooga/llama-tokenizer')

def compute_influences_for_file(input_file):
    instance_to_influences = pickle.load(open(input_file, "rb"))
    # two selection methods: min or mean or max
    # this is how we aggregate influences.
    # for mean, we just average scores across all test points.
    # for top min/max, we just take the min/max score across any test point.

    # map idx -> list of all influences
    all_train_scores = {}
    for i, influences in instance_to_influences.items():
        for train_idx, score in influences.items():
            if train_idx not in all_train_scores:
                all_train_scores[train_idx] = []
            all_train_scores[train_idx].append(score)
    
    # track the overall influences/idxes we are taking.
    overall_influences = {}
    if 'mean' in args.selection_method:
        print("Using mean influence selection method.")
        # mean reduce
        overall_influences = {k: mean(v) for k, v in all_train_scores.items()}
    elif 'min' in args.selection_method:
        print("Using top-min influence selection method.")
        # min reduce
        overall_influences = {k: min(v) for k, v in all_train_scores.items()}
    elif 'max' in args.selection_method:
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
    influence_scores = compute_influences_for_file(input_file)
    for idx, score in influence_scores.items():
        if idx not in per_dataset_influences:
            per_dataset_influences[idx] = []
        per_dataset_influences[idx].append(score)

# inter-dataset aggregation
inter_dataset_scores = {}
if args.aggregation_method == 'mean':
    print("Using mean inter-dataset aggregation.")
    aggregate_scores = {k: mean(v) for k, v in per_dataset_influences.items()}
elif args.aggregation_method == 'minmax':
    # base min/max on intra-dataset scores. TODO: more customizability here.
    print("Using minmax inter-dataset aggregation. Using selection method: {}".format(args.selection_method))
    fn = min if 'min' in args.selection_method else max
    aggregate_scores = {k: fn(v) for k, v in per_dataset_influences.items()}
else:
    raise ValueError("Invalid aggregation method.")

# finally, we sort and take topk. Again, base min/max on the selection method.
fn = sorted if 'min' in args.selection_method else lambda x: sorted(x, reverse=True)
aggregate_scores = fn(aggregate_scores.items(), key=lambda x: x[1])[:args.output_size]

# construct list of train indices
saved_instances = list(set([int(index) for index, _ in aggregate_scores]))
assert len(saved_instances) == args.output_size, "Saved instances size does not match output size."

# save the indices
with open(args.output_file, "w") as f:
    json.dump(saved_instances, f, indent=4)
