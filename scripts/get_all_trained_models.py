"""
quick script for getting all my old trained models.
"""
import argparse
from beaker import Beaker

parser = argparse.ArgumentParser(description="Get shard file for later processing")
parser.add_argument(
    "--prefix",
    type=str,
    default="rand_10k_logix_rank6_hesraw_fp32_unfiltered_tulu_shard_",
    help="prefix for the finetuned model pattern",
)
parser.add_argument("--outfile", type=str, help="output file for list")
parser.add_argument(
    "--workspace", type=str, default="ai2/minimal-multitask-finetuning", help="workspace to search for experiments"
)
args = parser.parse_args()

beaker = Beaker.from_env(default_workspace=args.workspace)

evals = ["alpacaeval", "alpacafarm", "gsm8k", "tydiqa", "bbh", "mmlu", "codex", "squad"]

# get all experiments in workspace matching the prefix
datasets = []
shards = []
for exp in beaker.workspace.iter_experiments(workspace=args.workspace, match=args.prefix):
    found_eval = False
    for eval in evals:
        if exp.name.endswith(eval):  # training runs end just with eval name, while evals runs provide extra info
            found_eval = True
            break
    if not found_eval:
        continue
    # get result
    result = beaker.experiment.results(exp)
    if result is None:
        continue
        raise ValueError(f"Experiment {exp} has no result")
    datasets.append(result.id)
    shards.append(exp.name)

# sort alphabetically
ds_and_shard = sorted(zip(datasets, shards), key=lambda x: x[1])

# write to file
with open(args.outfile, "a") as f:
    for ds, shard in ds_and_shard:
        f.write(f"{shard}\t{ds}\n")
    print(f"Wrote {len(ds_and_shard)} datasets to {args.outfile}")
