"""
quick script for getting shards txt file for later processing.
"""
import argparse
from beaker import Beaker

parser = argparse.ArgumentParser(description="Get shard file for later processing")
parser.add_argument(
    "--prefix",
    type=str,
    default="rand_10k_logix_rank6_hesraw_fp32_unfiltered_tulu_shard_",
    help="prefix for the experiment name",
)
parser.add_argument("--outfile", type=str, help="output file for shards")
parser.add_argument(
    "--workspace", type=str, default="ai2/minimal-multitask-finetuning", help="workspace to search for experiments"
)
args = parser.parse_args()

beaker = Beaker.from_env(default_workspace=args.workspace)

# get all experiments in workspace matching the prefix
datasets = []
shards = []
for exp in beaker.workspace.iter_experiments(workspace=args.workspace, match=args.prefix):
    # get result
    result = beaker.experiment.results(exp)
    if result is None:
        raise ValueError(f"Experiment {exp} has no result")
    datasets.append(result.id)
    shards.append(exp.name.split("_")[-1])

assert len(datasets) == len(shards), f"Mismatch between datasets and shards - got {len(datasets)}"
assert len(datasets) == 29, f"Expected 29 datasets - got {len(datasets)}"

# sort alphabetically
ds_and_shard = sorted(zip(datasets, shards), key=lambda x: x[1])

# write to file
with open(args.outfile, "w") as f:
    for ds, shard in ds_and_shard:
        f.write(f"{shard}\t{ds}\n")
    print(f"Wrote {len(ds_and_shard)} datasets to {args.outfile}")
