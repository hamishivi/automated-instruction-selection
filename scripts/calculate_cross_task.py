import argparse
import itertools
from collections import defaultdict
from statistics import mean
from typing import Any, Dict

from minimal_multitask.dataset_mapping import TASK_TO_PROMPTS

parser = argparse.ArgumentParser()
parser.add_argument(
    "--results_file",
    type=str,
    help="CSV from group",
)
parser.add_argument(
    "--task_list",
    default=None,
    type=str,
    help="Optional text file with list of tasks to examine. If not given we print values for all tasks.",
)
parser.add_argument(
    "--metric",
    default="rougeL",
    type=str,
    help="What metric to examine. Must be either rougeL or eval_loss",
)
args = parser.parse_args()

if args.task_list is None:
    task_list = list(TASK_TO_PROMPTS.keys())
else:
    task_list = [line.strip() for line in open(args.task_list, "r")]

task_list = sorted(task_list)
data = open(args.results_file, "r").readlines()

column_heads = data[0].split(",")

col_to_index = {}
for task in task_list:
    for i, col in enumerate(column_heads):
        if (
            task == col.replace(".eval_loss", "").replace("metrics_", "")
            and "eval_loss" == col.split(".")[-1]
        ):
            col_to_index[f"{task}\teval_loss"] = i
        if (
            task == col.replace(".eval_rougeL", "").replace("metrics_", "")
            and "eval_rougeL" == col.split(".")[-1]
        ):
            col_to_index[f"{task}\trougeL"] = i

task_results: Dict[Any, Any] = defaultdict(dict)
for line in data[1:]:
    data = line.split(",")
    experiment_name = data[9]
    for task, col_i in col_to_index.items():
        task, metric = task.split("\t")
        if args.metric == metric and data[col_i]:
            for task1, task2 in itertools.product(task_list, repeat=2):
                if task1 == task2 and "only" in experiment_name and task1 == task:
                    task_results[task1][task2] = float(data[col_i])
                elif (
                    task1 in experiment_name
                    and task2 in experiment_name
                    and task1 != task2
                    and task2 == task
                ):
                    task_results[task1][task2] = float(data[col_i])

for task1 in task_list:
    for task2 in task_list:
        if task2 in task_results[task1]:
            print(task_results[task1][task2], end=",")
        else:
            print("", end=",")
    print()

print("-" * 25)

# arg calculation
arg_scores: Dict[Any, Any] = defaultdict(dict)
for task1, task2 in itertools.product(task_list, repeat=2):
    # the cross has to exist
    if task2 not in task_results[task1] or task1 not in task_results[task2]:
        continue
    # if the single task doesnt exist, we cant calculate arg
    if task1 not in task_results[task1] or task2 not in task_results[task2]:
        continue
    task1_score = task_results[task2][task1]
    task2_score = task_results[task1][task2]
    base_score_task1 = task_results[task1][task1]
    base_score_task2 = task_results[task2][task2]
    task1_increase = (task1_score - base_score_task1) / base_score_task1
    task2_increase = (task2_score - base_score_task2) / base_score_task2
    arg_scores[task1][task2] = task2_increase
    arg_scores[task2][task1] = task1_increase

for task1 in task_list:
    for task2 in task_list:
        if task2 in arg_scores[task1]:
            print(arg_scores[task1][task2], end=",")
        else:
            print("", end=",")
    print()
