import argparse
import pickle
import os


parser = argparse.ArgumentParser()
parser.add_argument('--pickle_files', type=str, nargs='+', help='List of pickle files')
parser.add_argument('--save_dir', type=str, default='influence_scores/tulu2_unfiltered_llama2')
parser.add_argument('--file_name', type=str)

args = parser.parse_args()

combined_pkl = {}
pkl_max = 0
for fname in args.pickle_files:
    with open(fname, 'rb') as f:
        pkl = pickle.load(f)
    for k, v in pkl.items():
        if k not in combined_pkl:
            combined_pkl[k] = {}
        for idx, score in v.items():
            if idx < 0:
                continue
            else:
                combined_pkl[k][idx + pkl_max] = score
    pkl_max = max(combined_pkl[0].keys()) + 1
    print(f"Finish processing {fname}, cur_max_idx: {pkl_max}")

with open(os.path.join(args.save_dir, args.file_name), 'wb') as f:
    pickle.dump(combined_pkl, f)
