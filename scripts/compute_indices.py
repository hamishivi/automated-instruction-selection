'''
Given a selection file and a source file, get the indices of the selected lines in the source file.
'''
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Get indices of selected lines in source file')
parser.add_argument('selection_file', type=str, help='Selection file')
parser.add_argument('source_file', type=str, help='Source file')
parser.add_argument('output_file', type=str, help='Output file')
args = parser.parse_args()

selection = [json.loads(line.strip())['id'] for line in open(args.selection_file)]
indices = []

pbar = tqdm(total=len(selection))
for i, line in enumerate(open(args.source_file)):
    if json.loads(line.strip())['id'] in selection:
        indices.append(i)
        pbar.update(1)

with open(args.output_file, 'w') as f:
    for idx in indices:
        f.write(str(idx) + '\n')
