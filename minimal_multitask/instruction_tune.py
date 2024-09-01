import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
)
from minimal_multitask.utils import encode_with_messages_format
from typing import Optional
import os
import json
import sys
from peft import LoraConfig, TaskType, get_peft_model
from dataclasses import dataclass, field
from datasets import load_dataset, IterableDataset, Dataset


@dataclass
class AdditionalTrainingArguments:
    model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name: Optional[str] = field(default=None)
    train_dataset: str = field(
        default="alpaca",
        metadata={"help": "The dataset to train on. Can be Tulu2, LIMA, Alpaca, or a path to a jsonl file."},
    )
    lora_rank: Optional[int] = field(
        default=-1, metadata={"help": "The rank of the LoRA model. -1 means not using LoRA."}
    )
    lora_alpha: Optional[int] = field(
        default=-1, metadata={"help": "The rank of the LoRA model. -1 means not using LoRA."}
    )
    saved_instances: Optional[str] = field(
        default="", metadata={"help": "The optional file containing the indices of saved instances."}
    )
    random_select: Optional[int] = field(
        default=0, metadata={"help": "If set, randomly select the given number of instances from the train set."}
    )
    use_slow_tokenizer: Optional[bool] = field(default=False, metadata={"help": "Whether to use slow tokenizer."})
    use_flash_attention_2: Optional[bool] = field(default=True, metadata={"help": "Whether to use Flash Attention 2."})
    leak_test_data: Optional[bool] = field(
        default=False, metadata={"help": "Whether to leak test data into train data. Used for debugging."}
    )
    is_llama: Optional[bool] = field(
        default=True, metadata={"help": "If the current model is a llama model, used for LoRA wrapping."}
    )
    save_dir: Optional[str] = field(default="", metadata={"help": "The directory to save the model."})
    use_hf_auth_token: Optional[str] = field(default=False, metadata={"help": "Use the token stored in HF_TOKEN."})
    lora_ff_train: Optional[bool] = field(
        default=False, metadata={"help": "Whether to fully finetune the model along with the LoRA."}
    )


parser = HfArgumentParser((TrainingArguments, AdditionalTrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    trainer_args, additional_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    trainer_args, additional_args = parser.parse_args_into_dataclasses()

parser = HfArgumentParser((TrainingArguments, AdditionalTrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    trainer_args, additional_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    trainer_args, additional_args = parser.parse_args_into_dataclasses()

# model setup
kwargs = {}
if additional_args.use_flash_attention_2:
    kwargs["use_flash_attention_2"] = True
if additional_args.use_hf_auth_token is not None:
    kwargs["use_auth_token"] = os.environ.get("HF_TOKEN", None)
model = AutoModelForCausalLM.from_pretrained(
    additional_args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, **kwargs
)
if not additional_args.tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(
        additional_args.model_name, use_fast=not additional_args.use_slow_tokenizer, trust_remote_code=True, **kwargs
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(
        additional_args.tokenizer_name,
        use_fast=not additional_args.use_slow_tokenizer,
        trust_remote_code=True,
        **kwargs,
    )
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model.resize_token_embeddings(len(tokenizer))

# lora setup
if additional_args.lora_rank > -1:
    modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    if "llama" in additional_args.model_name or additional_args.is_llama:
        modules = ["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=additional_args.lora_rank,
        lora_alpha=additional_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=modules,
    )
    model = get_peft_model(model, peft_config)
    # if lora_ff_train is set, train all parameters, not just lora
    if additional_args.lora_ff_train:
        for name, param in model.named_parameters():
            param.requires_grad = True

# load and process train dataset
if additional_args.train_dataset == "alpaca":
    train_dataset = load_dataset("json", data_files="data/camel_datasets/stanford_alpaca/stanford_alpaca_data.jsonl")
    train_dataset = train_dataset["train"]
    train_dataset = train_dataset.map(lambda x: encode_with_messages_format(x, tokenizer, 1024, True, False))
elif additional_args.train_dataset == "lima":
    train_dataset = load_dataset("GAIR/lima", use_auth_token=True, split="train")

    def convert_lima(example):
        messages = [
            {"role": "user", "content": example["conversations"][0]},
            {"role": "assistant", "content": example["conversations"][1]},
        ]
        return {"messages": messages}

    train_dataset = train_dataset.map(convert_lima)
    train_dataset = train_dataset.map(lambda x: encode_with_messages_format(x, tokenizer, 1024, True, False))
elif additional_args.train_dataset == "tulu2":
    train_dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
    train_dataset = train_dataset.map(lambda x: encode_with_messages_format(x, tokenizer, 2048, True, False))
else:
    if os.path.exists(additional_args.train_dataset):
        # data files can be really big, but then we want to subselect
        train_dataset = load_dataset("json", data_files=additional_args.train_dataset, streaming=True)
        train_dataset = train_dataset["train"]
        train_dataset = train_dataset.map(lambda x: encode_with_messages_format(x, tokenizer, 2048, True, False))
    else:
        raise ValueError(f"Unknown dataset {additional_args.train_dataset}")

if type(train_dataset) is IterableDataset:
    train_dataset.with_format(type="torch")
else:
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
if additional_args.saved_instances != "":
    train_indices = json.load(open(additional_args.saved_instances, "r"))
    train_dataset = train_dataset.select(train_indices)
# for training, filter out empty instances
# do this after selection to ensure indices are consistent
train_dataset = train_dataset.filter(lambda x: (x["labels"] != -100).any())
if additional_args.random_select > 0:
    train_dataset = train_dataset.shuffle(seed=trainer_args.seed)
    if type(train_dataset) is IterableDataset:
        train_dataset = train_dataset.take(additional_args.random_select)
    else:
        train_dataset = train_dataset.select(range(additional_args.random_select))
    print(f"Randomly selected {additional_args.random_select} train instances")

# convert back to regular dataset, because trainer needs this.
if type(train_dataset) is IterableDataset:
    train_dataset = Dataset.from_generator(lambda: (yield from train_dataset), features=train_dataset.features)

# train on mix of train and test data
if additional_args.leak_test_data:
    from minimal_multitask.data import DATASETS

    test_dataset = DATASETS["squad"](tokenizer).get_all_test_prompts()
    train_dataset = train_dataset.select(range(len(test_dataset)))
    new_dataset = []
    for sample in test_dataset:
        train_dataset = train_dataset.add_item({k: v.tolist() for k, v in sample.items()})

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    args=trainer_args,
)
if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
trainer.train()
trainer.save_model(trainer_args.output_dir)
# if we are doing lora and ff training, then save the underlying model as well
if additional_args.lora_ff_train:
    # we have to unload to get the non-lora model (just getting the base doesnt work)
    underlying_model = model.unload()
    underlying_model.save_pretrained(os.path.join(trainer_args.output_dir, "underlying_model"))
