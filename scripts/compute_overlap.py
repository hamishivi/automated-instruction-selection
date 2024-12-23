'''
Compute number of overlapping samples between two jsonls
'''
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--file1', type=str, required=True)
parser.add_argument('--file2', type=str, required=True)
args = parser.parse_args()


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


load_data1 = load_jsonl(args.file1)
load_data2 = load_jsonl(args.file2)

print(f'Number of samples in file1: {len(load_data1)}')
print(f'Number of samples in file2: {len(load_data2)}')

# just grab ids and set
load_data1 = set([x['id'] for x in load_data1])
load_data2 = set([x['id'] for x in load_data2])

overlap = load_data1.intersection(load_data2)

# for overlapping ids, print out the influence scores in each file
# make id -> instance dict first for faster lookup
load_data1 = {x['id']: x for x in load_jsonl(args.file1)}
load_data2 = {x['id']: x for x in load_jsonl(args.file2)}
for id in overlap:
    print(f'Overlap id: {id}')
    print(f'File1: {load_data1[id]}')
    print(f'File2: {load_data2[id]}')
    print()

print(f'Number of overlapping samples: {len(overlap)}')
