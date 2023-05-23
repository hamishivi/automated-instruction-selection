import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import evaluate
import nltk
import numpy as np
from dataset_mapping import BBH_SUBSETS, TASK_TO_PROMPTS
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
    sample_file: Optional[str] = field(
        metadata={
            "help": "Path to file containing samples to train on. If given, overrides train_tasks."
        },
        default=None,
    )
    eval_bbh: bool = field(
        metadata={"help": "Whether to evaluate on the BBH dataset. If true, overrides eval_tasks."},
        default=False,
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
    max_source_length: int = field(
        default=768,
        metadata={"help": "Maximum length of inputs."},
    )
    max_target_length: int = field(
        default=256,
        metadata={"help": "Maximum length of outputs and generated text."},
    )


parser = HfArgumentParser((Seq2SeqTrainingArguments, DataArguments))
training_args, data_args = parser.parse_args_into_dataclasses()


train_tasks = data_args.train_tasks
eval_tasks = data_args.eval_tasks

if data_args.sample_file is not None:
    train_datasets = [
        load_dataset("json", data_files=data_args.sample_file).shuffle(seed=training_args.seed)[
            "train"
        ]
    ]
else:
    train_datasets = []
    for task in train_tasks:
        if task not in TASK_TO_PROMPTS:
            raise ValueError(
                f"train task {task} not valid. Tasks must be from {TASK_TO_PROMPTS.keys()}"
            )
        subprompts = TASK_TO_PROMPTS[task]
        subdatasets = []
        for prompt in subprompts:
            ds = load_dataset("bigscience/P3", prompt, split="train").shuffle(
                seed=training_args.seed
            )
            subdatasets.append(ds)
        # concatenate = size-proportional mixing.
        train_datasets.append(concatenate_datasets(subdatasets).shuffle(seed=training_args.seed))
        # cap at task level so we have roughly similar amounts of training data.
        if (
            data_args.max_samples_per_train_dataset > 0
            and len(train_datasets[-1]) > data_args.max_samples_per_train_dataset
        ):
            train_datasets[-1] = train_datasets[-1].select(
                range(data_args.max_samples_per_train_dataset)
            )


tokenizer_name = (
    data_args.tokenizer_name if data_args.tokenizer_name is not None else data_args.model_name
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSeq2SeqLM.from_pretrained(data_args.model_name)


if data_args.eval_bbh:
    subsets = BBH_SUBSETS
    prompts = [
        open(f"data/direct_bbh_prompts/{subset}.txt").read().split("-----")[-1]
        for subset in subsets
    ]
    eval_datasets = []
    eval_dataset_names = []
    # transform eval datasets to include prompts.
    for subset, prompt in zip(subsets, prompts):
        ds = load_dataset("lukaemon/bbh", subset, split="test")
        prompt = open(f"data/direct_bbh_prompts/{subset}.txt").read().split("-----")[1:]
        def prompt_data(batch):
            inputs = [f"\n{inp}\nAnswer: " for inp in batch["input"]]
            # dynamic adding of few-shot examples
            for i, inp in enumerate(inputs):
                few_shot_idx = 1
                while len(tokenizer(prompt[0] + inp)["input_ids"] + tokenizer(prompt[few_shot_idx])['input_ids']) < 2048:
                    inp = f"\n{prompt[few_shot_idx]}" + inp
                    few_shot_idx += 1
                    # stop at 3 shots
                    if few_shot_idx == len(prompt):
                        break
                batch["input"][i] = (prompt[0] + '\n' + inp).strip()
            return batch
        ds = ds.map(prompt_data, batched=True)
        # to match the format of the other datasets.
        ds = ds.rename_column("input", "inputs")
        ds = ds.rename_column("target", "targets")
        eval_datasets.append(ds)
        eval_dataset_names.append(subset)
else:
    eval_datasets = []
    eval_dataset_names = []
    for task in eval_tasks:
        if task not in TASK_TO_PROMPTS:
            raise ValueError(
                f"eval task {task} not valid. Tasks must be from {TASK_TO_PROMPTS.keys()}"
            )
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
            if (
                data_args.max_samples_per_eval_dataset > 0
                and len(ds) > data_args.max_samples_per_train_dataset
            ):
                ds = ds.select(range(data_args.max_samples_per_eval_dataset))
            eval_datasets.append(ds)
            eval_dataset_names.append(prompt)



# if sample file or bbh, we need to tokenize the data
def tokenize_function(examples):
    input_tokenized = tokenizer(
        examples["inputs"],
        truncation=True,
        return_tensors="pt",
        max_length=data_args.max_source_length,
        add_special_tokens=False,
    ).input_ids.long()
    target_tokenized = tokenizer(
        examples["targets"],
        truncation=True,
        return_tensors="pt",
        max_length=data_args.max_target_length,
        add_special_tokens=False,
    ).input_ids.long()
    return {"inputs": input_tokenized.flatten(), "targets": target_tokenized.flatten()}


if data_args.sample_file is not None:
    train_datasets = [ds.map(tokenize_function, num_proc=64) for ds in train_datasets]
if data_args.eval_bbh:
    eval_datasets = [ds.map(tokenize_function) for ds in eval_datasets]


# cut down lengths
# No eos on input matches how t0 was trained.
def preprocess_function(example):
    output = {"input_ids": example["inputs"]}
    if len(example["inputs"]) > data_args.max_source_length:
        output["input_ids"] = example["inputs"][: data_args.max_source_length - 1]
    output["input_ids"] += [tokenizer.eos_token_id]
    output["labels"] = example["targets"]
    if len(example["targets"]) > data_args.max_target_length:
        output["labels"] = example["targets"][: data_args.max_target_length - 1]
    output["labels"] += [tokenizer.eos_token_id]
    return output


def transform_ds(ds, num_proc=1):
    return ds.map(preprocess_function, num_proc=num_proc)


train_datasets = [transform_ds(ds, num_proc=64) for ds in train_datasets]
eval_datasets = [transform_ds(ds) for ds in eval_datasets]

# for some reason, flan has empty targets sometimes. filter them out.
train_datasets = [ds.filter(lambda x: len(x["labels"]) > 0) for ds in train_datasets]
eval_datasets = [ds.filter(lambda x: len(x["labels"]) > 0) for ds in eval_datasets]


nltk.download("punkt", quiet=True)
rouge = evaluate.load("rouge")
exact_match = evaluate.load("exact_match")


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # some light postprocessing for BBH.
    if data_args.eval_bbh:
        decoded_preds = [pred.strip()[:len(decoded_labels[i])] for i, pred in enumerate(decoded_preds)]
    exact_match_score = exact_match.compute(predictions=decoded_preds, references=decoded_labels)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    rouge_score = rouge.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    return {**rouge_score, **exact_match_score}

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

if training_args.do_train:
    print("Training model!")
    try:
        output = trainer.train(resume_from_checkpoint=training_args.output_dir)
    except ValueError:
        output = trainer.train()

if training_args.do_eval:
    print("Evaluating model!")
    metrics = trainer.evaluate(eval_datasets=eval_datasets, max_length=data_args.max_target_length)

    print("Postprocessing evaluation metrics!")
    counts: Dict[str, int] = {}
    averaged_metrics: Dict[str, float] = {}
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
                    averaged_metrics[f"{task}.{metric_name}"] = (
                        averaged_metrics.get(f"{task}.{metric_name}", 0) + value
                    )
                    counts[f"{task}.{metric_name}"] = counts.get(f"{task}.{metric_name}", 0) + 1

    metrics.update(averaged_metrics)

    # normalise metrics by number of subtasks
    for metric in metrics:
        metrics[metric] = metrics[metric] / counts.get(metric, 1)

    # save to metrics.json for beaker :)
    with open(data_args.metrics_output, "w") as w:
        json.dump(metrics, w)
