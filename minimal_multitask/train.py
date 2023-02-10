import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import evaluate
import nltk
import numpy as np
from dataset_mapping import TASK_TO_PROMPTS
from datasets import concatenate_datasets, interleave_datasets, load_dataset
from multi_eval_seq2seq_trainer import MultiEvalSeq2SeqTrainer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
)


@dataclass
class DataArguments:
    """
    Arguments about training, not covered by the seq2seq arguments
    """

    train_tasks: List[str] = field(
        default_factory=list,
        metadata={
            "help": f"List of the tasks to train on. Tasks must be from {TASK_TO_PROMPTS.keys()}."
        },
    )
    eval_tasks: List[str] = field(
        default_factory=list,
        metadata={
            "help": f"List of the tasks to evaluate on. Tasks must be from {TASK_TO_PROMPTS.keys()}."
        },
    )
    max_samples_per_train_dataset: int = field(
        default=10000,  # 10k as our 'reasonable default'. This covers the majority of task splits.
        metadata={"help": "Max instances to take from any train prompt."},
    )
    max_samples_per_eval_dataset: Optional[int] = field(
        default=-1, metadata={"help": "Max instances to take from any eval prompt."}
    )
    model_name: str = field(
        default="google/t5-xl-lm-adapt",
        metadata={"help": "Name of model. Must be a AutoModelForSeq2SeqLM-compatible model."},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of tokenizer to use. If not given, assume same name as model."},
    )
    metrics_output: str = field(
        default="metrics.json",
        metadata={"help": "Name of file to output metrics too. Default: metrics.json"},
    )


parser = HfArgumentParser((Seq2SeqTrainingArguments, DataArguments))
training_args, data_args = parser.parse_args_into_dataclasses()


train_tasks = data_args.train_tasks
eval_tasks = data_args.eval_tasks

train_datasets = []
for task in train_tasks:
    if task not in TASK_TO_PROMPTS:
        raise ValueError(
            f"train task {task} not valid. Tasks must be from {TASK_TO_PROMPTS.keys()}"
        )
    subprompts = TASK_TO_PROMPTS[task]
    subdatasets = []
    for prompt in subprompts:
        ds = load_dataset("bigscience/P3", prompt, split="train").shuffle(seed=training_args.seed)
        subdatasets.append(ds)
    # concatenate = size-proportional mixing.
    train_datasets.append(concatenate_datasets(subdatasets).shuffle(seed=training_args.seed))
    # cap at task level so we have roughly similar amounts of training data.
    if data_args.max_samples_per_train_dataset > 0:
        train_datasets[-1] = train_datasets[-1].select(
            range(data_args.max_samples_per_train_dataset)
        )

eval_datasets = []
eval_dataset_names = []
for task in eval_tasks:
    if task not in TASK_TO_PROMPTS:
        raise ValueError(f"eval task {task} not valid. Tasks must be from {TASK_TO_PROMPTS.keys()}")
    subprompts = TASK_TO_PROMPTS[task]
    for prompt in subprompts:
        ds = load_dataset("bigscience/P3", prompt)
        # annoyingly, not all datasets have validation sets.
        if "validation" in ds.keys():
            ds = ds["validation"]
        elif "test" in ds.keys():
            ds = ds["test"]
            print(f"{prompt} is using the test set for eval.")
        else:
            ds = ds["train"]
            print(f"{prompt} is using the train set for eval.")
        ds = ds.shuffle(seed=training_args.seed)
        if data_args.max_samples_per_eval_dataset > 0:
            ds = ds.select(range(data_args.max_samples_per_eval_dataset))
        eval_datasets.append(ds)
        eval_dataset_names.append(prompt)


# we have to remap the dataset to what t5 expects
# inputs -> input_ids
# targets -> labels
def transform_ds(ds):
    ds = ds.rename_column("inputs", "input_ids")
    ds = ds.rename_column("targets", "labels")
    return ds


train_datasets = [transform_ds(ds) for ds in train_datasets]
eval_datasets = [transform_ds(ds) for ds in eval_datasets]

tokenizer_name = (
    data_args.tokenizer_name if data_args.tokenizer_name is not None else data_args.model_name
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSeq2SeqLM.from_pretrained(data_args.model_name)

nltk.download("punkt", quiet=True)
metric = evaluate.load("rouge")


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return result


trainer = MultiEvalSeq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    ),
    train_dataset=interleave_datasets(train_datasets, seed=training_args.seed),
    eval_datasets=eval_datasets,
    eval_dataset=eval_datasets,
    eval_dataset_names=eval_dataset_names,
    compute_metrics=compute_metrics,
)

print("Training model!")
# output = trainer.train()

print("Evaluating model!")
metrics = trainer.evaluate(eval_datasets=eval_datasets)

print("Postprocessing evaluation metrics!")
counts: Dict[str, int] = {}
averaged_metrics = {}
# final postprocessing: average across prompts from the same task, weighted by size.
for task in eval_tasks:
    subprompts = TASK_TO_PROMPTS[task]
    for metric in metrics:
        for prompt in subprompts:
            # some prompts can be substrings of others, but we know the metric
            # uses format <prompt>.<metric>. Metric names don't contain '.' but
            # prompts can, hence splitting and rejoining.
            if prompt.lower() == ".".join(metric.lower().split(".")[:-1]):
                metric_name = metric.split(".")[-1]
                value = metrics[metric]
                averaged_metrics[f"{task}.{metric_name}"] = averaged_metrics.get(f"{task}.{metric_name}", 0) + value
                counts[f"{task}.{metric_name}"] = counts.get(f"{task}.{metric_name}", 0) + 1

metrics.update(averaged_metrics)

# normalise metrics by number of subtasks
for metric in metrics:
    metrics[metric] = metrics[metric] / counts.get(metric, 1)

# save to metrics.json for beaker :)
with open(data_args.metrics_output, "w") as w:
    json.dump(metrics, w)
