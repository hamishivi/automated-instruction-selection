'''
Script for combining sharded pickles into one big pickle.
Useful for e.g. analysis scripts.
'''
import argparse
import pickle
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_files", type=str, nargs="+")  # we can pass in multiple pickles
parser.add_argument("--output_file", type=str)  # where to save the pickle to.
args = parser.parse_args()

# sort so i dont have to worry about order
args.input_files = sorted([f for f in args.input_files if f != args.output_file])

# load the pickles
influences = [pickle.load(open(f, "rb")) for f in tqdm(args.input_files)]

# combine the pickles
combined_influence = {}
ongoing_counter = 0
for influence in tqdm(influences):
    for test_idx in influence:
        if test_idx not in combined_influence:
            combined_influence[test_idx] = {}
        for train_idx, score in influence[test_idx].items():
            # ignore bogus index.
            if train_idx == -1:
                skip = True
                continue
            if not type(score) is float:
                score = float(score.item())
            if not type(test_idx) is int:
                test_idx = int(test_idx.item())
            if not type(train_idx) is int:
                train_idx = int(train_idx.item())
            combined_influence[test_idx][train_idx + ongoing_counter] = score
    # adjust for the fact we skipped one if we did.
    ongoing_counter += len(influence[0])
    if skip:
        ongoing_counter -= 1
        skip = False

# save as a json for space reasons
json.dump(combined_influence, open(args.output_file.replace(".pkl", ".json"), "w"))
print(f"Saved combined pickle to {args.output_file}")
