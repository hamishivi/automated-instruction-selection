'''
A script for quickly collecting beaker results given a prefix.
Useful for quickly collecting results to paste into a spreadsheet.
'''
import argparse
from beaker import Beaker

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", type=str, required=True)
parser.add_argument(
    "--workspace", type=str, default="ai2/minimal-multitask-finetuning", help="workspace to search for experiments"
)
args = parser.parse_args()

# some nice things
EVAL_NAMES = ["MMLU", "GSM8k", "BBH", "TydiQA", "Codex", "Squad", "AlpacaEval"]
EVAL_METRICS = ["average_acc", "exact_match", "average_exact_match", "average:f1", "pass@10", "f1", "win_rate"]
EVAL_JOB_NAMES = {
    "mmlu_0shot": "MMLU",
    "gsm_cot": "GSM8k",
    "bbh_cot": "BBH",
    "tydiqa_goldp": "TydiQA",
    "codex_pass10": "Codex",
    "squad_context": "Squad",
    "alpaca_eval": "AlpacaEval",
}
EVAL_NAME_TO_METRIC = {name: metric_name for name, metric_name in zip(EVAL_NAMES, EVAL_METRICS)}

beaker = Beaker.from_env(default_workspace=args.workspace)

# get all experiments in workspace matching the prefix
results = {}
for exp in beaker.workspace.iter_experiments(workspace=args.workspace, match=args.prefix):
    # grab eval name
    eval_name = None
    for name in EVAL_JOB_NAMES.keys():
        if name in exp.name:
            eval_name = name
            break
    if eval_name is None:
        print(f"Experiment {exp.name} does not have a recognized eval name.")
        continue
    # grab metric
    metrics = beaker.experiment.metrics(exp)
    if metrics is None:
        print(f"Experiment {exp.name} has no metrics. Maybe waiting for results?")
        continue
    # put results in dict
    metric_name = EVAL_NAME_TO_METRIC[EVAL_JOB_NAMES[eval_name]].split(":")
    if len(metric_name) == 1:
        results[EVAL_JOB_NAMES[eval_name]] = metrics[metric_name[0]]
    else:
        # currently only other case is tydiqa, so just hardcode
        results[EVAL_JOB_NAMES[eval_name]] = metrics["average"]["f1"]

# finally, print results in order given with tab spacing
for eval_name in EVAL_NAMES:
    print(eval_name, end='\t')
print()
for eval_name in EVAL_NAMES:
    if eval_name in results:
        print(results[eval_name], end='\t')
    else:
        print("N/A", end='\t')
print()
