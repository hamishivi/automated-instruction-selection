import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from minimal_multitask.utils import encode_with_messages_format

from tqdm import tqdm
import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--tokenizer", type=str, default=None)
parser.add_argument("--save_dir", type=str, default="selections/ppl_200k_exp")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--train_dataset", type=str, default="alpaca")
parser.add_argument("--index_path", type=str)
parser.add_argument("--dtype", default="bf16")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--prompt_only", action="store_true")
parser.add_argument("--label_only", action="store_true")
parser.add_argument("--only_first_two", action="store_true")  # only use the first two messages
args = parser.parse_args()

torch.manual_seed(args.seed)
if args.dtype == "bf16":
    kwargs = {"torch_dtype": torch.bfloat16}
elif args.dtype == "fp16":
    kwargs = {"torch_dtype": torch.float16}
elif args.dtype == "fp32":
    kwargs = {"torch_dtype": torch.float32}
if "llama" in args.model_name:
    kwargs["attn_implementation"] = "sdpa"

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    **kwargs,
    device_map="auto",  # use multiple gpus if you can
)

if args.tokenizer is not None:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# load and process train dataset
if args.train_dataset == "alpaca":
    train_dataset = load_dataset("json", data_files="data/stanford_alpaca_data.jsonl")[
        "train"
    ]
    train_dataset = train_dataset.map(
        lambda x: encode_with_messages_format(x, tokenizer, 512, True, args.label_only, args.only_first_two, args.prompt_only), num_proc=16
    )
elif args.train_dataset == "tulu2":
    train_dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
    train_dataset = train_dataset.map(
        lambda x: encode_with_messages_format(x, tokenizer, 2048, True, args.label_only, args.only_first_two, args.prompt_only), num_proc=16
    )
else:
    if os.path.exists(args.train_dataset):
        train_dataset = load_dataset("json", data_files=args.train_dataset)["train"]
        train_dataset = train_dataset.map(
            lambda x: encode_with_messages_format(x, tokenizer, 2048, True, args.label_only, args.only_first_two, args.prompt_only), num_proc=1, load_from_cache_file=True, keep_in_memory=False
        )
    else:
        raise ValueError(f"Invalid train dataset: {args.train_dataset}")
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

print(f"Train dataset size: {len(train_dataset)}")
# Compute the perplexity on each sample
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
nlls = {}

model.eval()

with torch.no_grad():
    for idx, batch in enumerate(tqdm(dataloader)):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        nlls[idx] = loss.item()

with open(os.path.join(args.save_dir, "nlls.pkl"), "wb") as f:
    pickle.dump(nlls, f)
