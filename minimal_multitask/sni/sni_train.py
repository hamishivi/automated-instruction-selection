import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset
from sni_collator import DataCollatorForNI
from sni_evaluation import compute_all_metrics, compute_metrics
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


@dataclass
class DataArguments:
    """
    Arguments about training, not covered by the seq2seq arguments
    """

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
    num_pos_examples: int = field(
        default=0,
        metadata={"help": "Number of positive examples to use. Usually 0 or 2, 0 by default."},
    )
    reference_file: str = field(
        default="/net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/minimal_multitask/sni/test_references.jsonl",  # noqa
        metadata={"help": "SNI eval reference file."},
    )


parser = HfArgumentParser((Seq2SeqTrainingArguments, DataArguments))
training_args, data_args = parser.parse_args_into_dataclasses()


train_dataset = load_dataset("minimal_multitask/sni/sni_dataset.py", split="train")
eval_dataset = load_dataset("minimal_multitask/sni/sni_dataset.py", split="test")

tokenizer_name = (
    data_args.tokenizer_name if data_args.tokenizer_name is not None else data_args.model_name
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSeq2SeqLM.from_pretrained(data_args.model_name)


data_collator = DataCollatorForNI(
    tokenizer,
    num_pos_examples=data_args.num_pos_examples,
    max_source_length=data_args.max_source_length,
    max_target_length=data_args.max_target_length,
)


def transform_ds(sample):
    res = data_collator(sample)
    return {k: v.long().flatten().tolist() for k, v in res.items()}


train_dataset = train_dataset.map(transform_ds, batched=False, num_proc=32)
eval_dataset = eval_dataset.map(transform_ds, batched=False, num_proc=32)


def metrics_wrapper(eval_pred):
    predictions, labels = eval_pred
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # formatting thing for metrics
    labels = [[label] for label in labels]
    return compute_metrics(predictions, labels)


trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=metrics_wrapper,
)

if training_args.do_train:
    print("Training model!")
    try:
        output = trainer.train(resume_from_checkpoint=training_args.output_dir)
    except ValueError:
        output = trainer.train()

if training_args.do_eval:
    predictions, _, metrics = trainer.predict(eval_dataset)
    results = {
        id: tokenizer.decode(pred, skip_special_tokens=True)
        for id, pred in zip(eval_dataset["id"], predictions)
    }

    eval_instances = {}
    with open(data_args.reference_file) as fin:
        for line in fin:
            instance = json.loads(line)
            # if track is not provided in the refernce file, we use set the track
            # to `default` and use the default tokenizer in rouge-score.
            if "track" not in instance:
                instance["track"] = "default"
            eval_instances[instance["id"]] = instance

    compute_all_metrics(results, eval_instances, data_args.metrics_output)
