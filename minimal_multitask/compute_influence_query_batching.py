import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from minimal_multitask.nn_influence_utils import compute_influences_batched
from minimal_multitask.utils import encode_with_messages_format
from datasets import load_dataset
import argparse
import pickle
from minimal_multitask.data import DATASETS

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m")
parser.add_argument("--tokenizer", type=str, default=None)
parser.add_argument("--use_fjlt", action="store_true")
parser.add_argument("--top_k", type=int, default=100)
parser.add_argument("--instance_to_influences", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--eval_dataset", type=str, choices=DATASETS.keys(), default="mmlu")
parser.add_argument("--index_path", type=str)
args = parser.parse_args()

torch.manual_seed(args.seed)

# model loading
kwargs = {"torch_dtype": torch.bfloat16}
if "llama" in args.model_name:
    kwargs["use_flash_attention_2"] = True


model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    **kwargs,
    device_map="auto",  # use multiple gpus if you can
)
# loading sets requires_grad to False, so we need to set it back to True
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True

params_filter = [n for n, p in model.named_parameters() if not p.requires_grad]

weight_decay_ignores = ["bias", "LayerNorm.weight"] + [n for n, p in model.named_parameters() if not p.requires_grad]


if args.tokenizer is not None:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# load and process train dataset
train_dataset = load_dataset("json", data_files="data/camel_datasets/stanford_alpaca/stanford_alpaca_data.jsonl")
train_dataset = train_dataset.map(lambda x: encode_with_messages_format(x, tokenizer, 512, True, False))
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
train_dataset = train_dataset["train"]

# test dataset - mostly handled in data.py
if args.eval_dataset in DATASETS:
    test_dataset = DATASETS[args.eval_dataset](tokenizer).get_all_test_prompts()
else:
    raise ValueError(f"Invalid dataset: {args.dataset}")


print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
# construct dataloaders
batch_train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
instance_train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
eval_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# influence calculations. This script does query batching, like anthropic paper.
# test_batches is used to control batch size when computing influence. Reduce if getting OOM.
instance_to_influences = {}
influences, topk_indices, _ = compute_influences_batched(
    n_gpu=1,
    device=torch.device("cuda"),
    model=model,
    test_inputs=eval_data_loader,
    batch_train_data_loader=batch_train_data_loader,
    instance_train_data_loader=instance_train_data_loader,
    params_filter=params_filter,
    weight_decay=0.0,
    weight_decay_ignores=weight_decay_ignores,
    s_test_damp=5e-3,
    s_test_scale=1e6,
    s_test_num_samples=100,
    s_test_iterations=1,
    precomputed_s_test=None,
    test_batches=2,
)
# save results for post processing later.
for test_idx, infs in enumerate(influences):
    instance_to_influences[test_idx] = {}
    for train_index in infs:
        instance_to_influences[test_idx][train_index] = infs[train_index]

with open(args.instance_to_influences, "wb") as f:
    pickle.dump(instance_to_influences, f)
