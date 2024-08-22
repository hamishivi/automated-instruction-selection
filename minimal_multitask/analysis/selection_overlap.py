import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--selection_results", type=str, nargs="+")
parser.add_argument("--topk", type=int)

args = parser.parse_args()

for idx, result_a in enumerate(args.selection_results):
    if idx == len(args.selection_results) - 1:
        break
    with open(result_a, "r") as f:
        index_a = json.load(f)
        if args.topk:
            index_a = index_a[: args.topk]

    for idx_b, result_b in enumerate(args.selection_results[idx + 1:]):
        with open(result_b, "r") as f:
            index_b = json.load(f)
            if args.topk:
                index_b = index_b[: args.topk]

        overlap = len(set(index_a).intersection(set(index_b)))
        print(f"Overlap between {result_a} and {result_b}: {overlap}")
