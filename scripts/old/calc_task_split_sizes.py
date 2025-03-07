from typing import Dict

from scripts.old.split_sizes import DATA_SPLITS_SIZES

from minimal_multitask.dataset_mapping import TASK_TO_PROMPTS

task_split_sizes = {}

for task, prompts in TASK_TO_PROMPTS.items():
    split_dict: Dict[str, int] = {}
    for p in prompts:
        splits = DATA_SPLITS_SIZES[p]
        for split, size in splits.items():
            split_dict[split] = split_dict.get(split, 0) + size
    task_split_sizes[task] = split_dict

train_split_sizes = []
for task, splits in task_split_sizes.items():
    train_split_sizes.append(splits["train"])

print(f"Min validation split size: {sorted(train_split_sizes)}")
