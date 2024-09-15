'''
Quickly count subset counts in jsonl or hf dataset
'''
import json
from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser(description='Count subset counts in jsonl or hf dataset')
parser.add_argument('--path', type=str, help='path to jsonl file or hf dataset')
args = parser.parse_args()

dataset = []
if args.path.endswith('.jsonl'):
    with open(args.path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
else:
    dataset = load_dataset(args.path, split='train')

# count 'dataset' key unique values
counts = {}
for item in dataset:
    key = item['dataset']
    if key not in counts:
        counts[key] = 0
    counts[key] += 1

for k, v in counts.items():
    print(f'{k}: {v}')
