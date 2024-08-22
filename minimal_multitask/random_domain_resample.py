import os
import argparse
import json
import random
import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--output_file", type=str)
parser.add_argument("--output_size", type=int, default=10000)  # number of instances total to select.
parser.add_argument("--train_datasets", nargs="+", type=str, default=["alpaca"])  # alpaca, tulu2
parser.add_argument("--domain_weights", type=str)  # json file containing domain weights normalized to 1.
args = parser.parse_args()

random_gen = random.Random(42)

# for cot data
flan_mapping = {}
flan_filename = "flanv2-1M-v2.jsonl"
for line in tqdm.tqdm(open(flan_filename, "r")):
    data = json.loads(line)
    info = json.loads(data["info"])
    prompt = data["inputs"]
    # match formatting step we did
    if not prompt.endswith("\n") and not prompt.rstrip().endswith(":"):
        prompt += "\n"
    flan_mapping[data["inputs"]] = info["_task_name"]


# load domain weight information
if args.domain_weights:
    domain_weights = json.load(open(args.domain_weights))
    # normalize domain weights just in case
    domain_weights = {k: v / sum(domain_weights.values()) for k, v in domain_weights.items()}
    # domain max size
    domain_max_size = {k: v * args.output_size for k, v in domain_weights.items()}
else:
    domain_max_size = None


def get_domain_name(domain, input):
    if "science" in domain:
        return "science"
    # parse out cot samples
    if "flan" in domain:
        subtask = flan_mapping.get(input, domain)
        if "cot" in subtask:
            return "cot"
    return domain


def get_domain_values(domain):
    if "science" in domain and "science" in domain_max_size:
        return domain_max_size["science"]
    elif domain not in domain_max_size:
        return 0
    return domain_max_size[domain]


# load train datasets for printing
train_datasets = []
for train_dataset in args.train_datasets:
    if train_dataset == "alpaca":
        train_datasets.append(
            load_dataset("json", data_files="data/camel_datasets/stanford_alpaca/stanford_alpaca_data.jsonl")["train"]
        )
    elif train_dataset == "tulu2":
        train_datasets.append(load_dataset("allenai/tulu-v2-sft-mixture", split="train"))
    else:
        # assume it's a path to a dataset
        if os.path.exists(train_dataset):
            train_datasets.append(load_dataset("json", data_files=train_dataset)["train"])
        else:
            raise ValueError(f"Invalid train dataset {train_dataset}.")
# just assume llama tokenizer for now. This is used for debugging mainly.
tokenizer = AutoTokenizer.from_pretrained("oobabooga/llama-tokenizer")

# collate all train datasets into flattened list
all_train_datasets = []
if len(train_datasets) == 1:
    all_train_datasets = train_datasets[0].shuffle(seed=42)
else:
    raise ValueError("Only one train dataset supported for now.")

# for each domain, sample the domain size
output_dataset = []
for domain in domain_max_size.keys():
    domain_size = int(get_domain_values(domain))
    if domain_size == 0:
        continue
    # in to deal with science.
    if domain == "cot":

        def is_cot(sample):
            if sample["dataset"] == "open_orca":
                return False
            return "cot" in flan_mapping.get(sample["messages"][0]["content"].strip(), "")

        domain_dataset = all_train_datasets.filter(is_cot, num_proc=32)
        # rename domain to cot
        domain_dataset = domain_dataset.map(lambda x: {"dataset": "cot", "messages": x["messages"]}, num_proc=32)
    else:
        domain_dataset = all_train_datasets.filter(lambda x: domain in x["dataset"], num_proc=32)
    sampled_domain_dataset = domain_dataset.shuffle(seed=42).select(list(range(domain_size)))
    output_dataset.extend(sampled_domain_dataset)
    print(f"Sampled {len(sampled_domain_dataset)} instances from domain {domain}.")

# assert len(output_dataset) == args.output_size, f"Output dataset size {len(output_dataset)} does not match expected size {args.output_size}."

# save the output dataset in jsonl
with open(args.output_file, "w") as f:
    for instance in output_dataset:
        f.write(json.dumps(instance) + "\n")
