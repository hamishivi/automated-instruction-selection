import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scripts.create_llama_encodings import encode_with_messages_format
from datasets import load_dataset
import argparse
from tqdm import tqdm
import pickle
from collections import defaultdict
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
instance_train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
eval_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

model.eval()

# current attempt at datainf computation, based on https://github.com/ykwon0407/DataInf/blob/main/src/influence.py
# note that this currently holds all gradients in memory, and so is probably quite slow.
# I will work on speeding it up if I can see it working well.


def get_gradient(sample):
    model.zero_grad()
    loss = model(**sample).loss
    loss.backward()
    return {n: p.grad.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


tr_grad_dict = {}
for index, train_instance in tqdm(enumerate(instance_train_data_loader)):
    grads = get_gradient(train_instance)
    for k in grads:
        if "B" in k:
            grads[k] = grads[k].T
    tr_grad_dict[index] = grads

val_grad_dict = {}
for index, val_instance in tqdm(enumerate(eval_data_loader)):
    grads = get_gradient(val_instance)
    for k in grads:
        if "B" in k:
            grads[k] = grads[k].T
    val_grad_dict[index] = grads

# save gradients
with open("alpaca_tr_grad_dict.pkl", "wb") as f:
    pickle.dump(tr_grad_dict, f)

with open(args.eval_dataset + "_val_grad_dict.pkl", "wb") as f:
    pickle.dump(val_grad_dict, f)

hvp_proposed_dict = defaultdict(dict)
lambda_const_param = 10
influence_scores = {}
for val_id in tqdm(val_grad_dict.keys()):
    for weight_name in val_grad_dict[val_id]:
        # lambda_const computation
        S = torch.zeros(len(tr_grad_dict.keys()))
        for tr_id in tr_grad_dict:
            tmp_grad = tr_grad_dict[tr_id][weight_name]
            S[tr_id] = torch.mean(tmp_grad**2)
        lambda_const = torch.mean(S) / lambda_const_param  # layer-wise lambda
        # hvp computation
        hvp = torch.zeros(val_grad_dict[val_id][weight_name].shape)
        for tr_id in tr_grad_dict:
            C_tmp = torch.sum(val_grad_dict[val_id][weight_name] * tmp_grad) / (
                lambda_const + torch.sum(tmp_grad**2)
            )
            hvp += (val_grad_dict[val_id][weight_name] - C_tmp * tmp_grad) / (len(tr_grad_dict) * lambda_const)
        hvp_proposed_dict[val_id][weight_name] = hvp
    # influence score computation
    influence_scores[val_id] = {}
    for tr_id in tr_grad_dict:
        if_tmp_val = 0
        for weight_name in val_grad_dict[0]:
            if_tmp_val += torch.sum(hvp_proposed_dict[val_id][weight_name] * tr_grad_dict[tr_id][weight_name])
        influence_scores[val_id][tr_id] = if_tmp_val

# save
with open(args.instance_to_influences, "wb") as f:
    pickle.dump(influence_scores, f)

print(f"Saved to {args.instance_to_influences}")
