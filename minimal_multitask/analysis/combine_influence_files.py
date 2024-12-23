import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--pickle_files", type=str, nargs="+")
parser.add_argument("--save_dir", default="results/llama_2_tulu2_unfiltered_multiround")
parser.add_argument("--save_file_name", default="combined_influence")

args = parser.parse_args()

new_pickle = {}
cur_max_idx = 0
num_zero_score_total = 0
for file_name in args.pickle_files:
    pickle_file = pickle.load(open(file_name, "rb"))
    num_excluded_indices = 0
    num_zero_score_per_file = 0
    for index in pickle_file.keys():
        if index not in new_pickle:
            new_pickle[index] = {}
        inf_scores = pickle_file[index]
        for idx, score in inf_scores.items():
            if idx == -1:
                num_excluded_indices = 1
                continue
            if score == 0:
                num_zero_score_per_file += 1
            new_pickle[index][cur_max_idx + idx] = score

    cur_max_idx += len(inf_scores) - num_excluded_indices
    print(f"Finish processing {file_name}, cur_max_idx: {cur_max_idx}")
    print(f"Number of zero influence score in {file_name}: {num_zero_score_per_file}")
    num_zero_score_total += num_zero_score_per_file

with open(os.path.join(args.save_dir, f"{args.save_file_name}.pkl"), "wb") as fout:
    pickle.dump(new_pickle, fout)
