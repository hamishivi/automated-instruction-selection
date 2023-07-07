"""
Script to sample from a file while restricting to tasks only in a given list
"""
import argparse
import json
import random
from typing import Dict, Set

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--task_list", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--sample_size", type=int, default=25000)
parser.add_argument("--task_percentage", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random_gen = random.Random(args.seed)

task_list = [t.strip() for t in open(args.task_list, "r").readlines()]
# shrink the task list. We do this here to sample different tasks each time.
random_gen.shuffle(task_list)
task_list = task_list[: int(len(task_list) * args.task_percentage)]
# get samples
samples = [json.loads(x.strip()) for x in tqdm(open(args.input_file, "r").readlines())]
# filter - info is a stringified json so we need to parse it again.
filtered_samples = [x for x in tqdm(samples) if json.loads(x["info"])["_task_name"] in task_list]
print(f"Filtered {len(samples)} samples to {len(filtered_samples)} samples")
# to save memory
del samples

# choices as we want to sample without replacement
samples = random_gen.sample(filtered_samples, k=args.sample_size)
seen_samples: Set[Dict] = set()
with open(args.output_file, "w") as w:
    for sample in tqdm(samples):
        w.write(json.dumps(sample) + "\n")
