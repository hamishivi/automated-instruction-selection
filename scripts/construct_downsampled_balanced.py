'''
Construct the downsampled balanced dataset used for analysis experiments.
'''
import argparse
import json
from tqdm import tqdm
import random

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, help='Input jsonl file')
parser.add_argument('--output_file', type=str, help='Output jsonl file')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--num_samples', type=int, default=200_000, help='Number of samples to downsample to')
args = parser.parse_args()

random.seed(args.seed)

data = []
with open(args.input_file) as f:
    for line in tqdm(f):
        data.append(json.loads(line))

assert args.num_samples <= len(data), f'Cannot sample more than {len(data)} samples'

# separate data by dataset
datasets = {}
for d in tqdm(data):
    dataset = d['dataset']
    if dataset not in datasets:
        datasets[dataset] = []
    datasets[dataset].append(d)

# collapse science together
science_datasets = [k for k in datasets.keys() if 'science' in k]
science_data = []
for k in science_datasets:
    science_data.extend(datasets.pop(k))
datasets['science'] = science_data
# remove hardcoded dataset
datasets.pop('hard_coded', None)

# print the length of each dataset
print('Dataset distribution:')
for dataset, dataset_data in datasets.items():
    print(f'{dataset}: {len(dataset_data)}')


# randomly sample evenly from each dataset until we reach num_samples
sampled_data = []
pbar = tqdm(total=args.num_samples)
# round robin sample to be as even as possible, but fill to max
while len(sampled_data) < args.num_samples:
    for dataset, dataset_data in datasets.items():
        if len(dataset_data) == 0:
            continue
        if len(sampled_data) >= args.num_samples:
            break
        sampled_data.append(dataset_data.pop(0))
        pbar.update(1)

assert len(sampled_data) == args.num_samples, f'Expected {args.num_samples} samples, got {len(sampled_data)}'

# shuffle it to be safe
random.shuffle(sampled_data)

# write to output file
with open(args.output_file, 'w') as f:
    for d in sampled_data:
        f.write(json.dumps(d) + '\n')

# print some stats
print(f'Sampled {args.num_samples} samples from {args.input_file} and saved to {args.output_file}')
print('Dataset distribution:')
for dataset, dataset_data in datasets.items():
    # 'in' is because of the science <-> real dataset name mismatch
    print(f'{dataset}: {len([d for d in sampled_data if dataset in d["dataset"]])}')
