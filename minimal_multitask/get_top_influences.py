import pickle
import json
from datasets import load_dataset
import argparse
from collections import defaultdict
from statistics import mean

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default=None)
parser.add_argument('--output_file', type=str, default=None)
parser.add_argument('--filter_average_influence', type=float, default=None) # we will negate this value, note.
parser.add_argument('--take_top_k', type=int, default=1000) # take topk per instance
args = parser.parse_args()

# instance info
instance_to_influences = pickle.load(open(args.input_file, "rb"))

# load train dataset for printing
# train_dataset = load_dataset('json', data_files='data/camel_datasets/stanford_alpaca/stanford_alpaca_data.jsonl')['train']

# if we want to filter by average influence, we first compute the average influence for each instance
if args.filter_average_influence is not None:
    all_influences = defaultdict(list)
    for i, influences in instance_to_influences.items():
        for index, influence in influences.items():
            all_influences[index].append(influence)
    average_influences = {k: mean(v) for k, v in all_influences.items()}
else:
    average_influences = None

# for each test instance, lets take the top-10 most influential train instances
saved_instances = []
for i, influences in instance_to_influences.items():
    index_list, influence_list = influences.keys(), influences.values()
    if average_influences is not None:
        # < since 'good' influence is high negative.
        index_list = [index for index in index_list if average_influences[index] < args.filter_average_influence]
        influence_list = [influence for index, influence in zip(index_list, influence_list) if average_influences[index] < -args.filter_average_influence]
    top_influences = sorted(zip(influence_list, index_list), reverse=True)
    # optionally just take top k. Allows closer control of how many instances we train over.
    if args.take_top_k is not None:
        top_influences = top_influences[:args.take_top_k]
    # clunky if but there in case we have used some different code
    saved_instances += [index if isinstance(index, int) else index.item() for _, index in top_influences]

saved_instances = list(set(saved_instances))
print(f"Saved {len(saved_instances)} instances")
# save top instances
if not args.output_file:
    args.output_file = f"{args.input_file.split('.')[0]}_top{args.take_top_k}.json"
with open(args.output_file, "w") as f:
    json.dump(saved_instances, f, indent=4)

