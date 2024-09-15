import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--selection_files", type=str, nargs="+")
parser.add_argument("--output_file", type=str)
parser.add_argument("--deduplication", action='store_true')

args = parser.parse_args()

combined_json = []
for fname in args.selection_files:
    with open(fname, 'r') as f:
        for line in f:
            combined_json.append(json.loads(line))
print(f"# of samples before deduplication: {len(combined_json)}")
    
if args.deduplication:
    unique_samples = {sample["id"]: sample for sample in combined_json}.values()
    combined_json = list(unique_samples)
    print(f"# of samples after deduplication: {len(combined_json)}")
    
with open(args.output_file, 'w') as f:
    json.dump(combined_json, f)