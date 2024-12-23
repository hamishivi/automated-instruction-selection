import argparse
import pickle
import json
import gc
from tqdm import tqdm


def process_pickle_file(pickle_file, combined_influence, ongoing_counter):
    with open(pickle_file, "rb") as f:
        influence = pickle.load(f)

    skip = False
    for test_idx, train_scores in influence.items():
        if test_idx not in combined_influence:
            combined_influence[test_idx] = {}

        for train_idx, score in train_scores.items():
            # Ignore invalid index
            if train_idx == -1:
                skip = True
                continue

            train_idx = int(train_idx) + ongoing_counter
            test_idx = int(test_idx)
            score = float(score)

            # Update combined influence
            combined_influence[test_idx][train_idx] = score

    ongoing_counter += len(influence[0])
    if skip:
        ongoing_counter -= 1

    # free the memory
    del influence
    gc.collect()
    return ongoing_counter


def main(args):
    input_files = sorted([f for f in args.input_files if f != args.output_file])

    combined_influence = {}
    ongoing_counter = 0

    # Process each pickle independently
    for pickle_file in tqdm(input_files, desc="Processing Pickles"):
        ongoing_counter = process_pickle_file(pickle_file, combined_influence, ongoing_counter)

    output_json = args.output_file.replace(".pkl", ".json")
    with open(output_json, "w") as f:
        json.dump(combined_influence, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, nargs="+")
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    main(args)
