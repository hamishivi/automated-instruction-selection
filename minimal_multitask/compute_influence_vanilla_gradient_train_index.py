import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scripts.create_llama_encodings import encode_with_messages_format
from datasets import load_dataset
import argparse
import faiss
from tqdm import tqdm
import os
import pickle
from minimal_multitask.nn_influence_utils import compute_gradients
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

# construct an index over the gradients on the train data
# use inner product.
num_params = sum([p.numel() for n, p in model.named_parameters() if p.requires_grad])
grad_index = faiss.index_factory(num_params, "Flat", faiss.METRIC_INNER_PRODUCT)

# we add to the index in batches to speed things up?
grad_batch = 512
accum_grads = []
# save gradients for visualisation later.
samples = []
counter = 0
if not os.path.exists(args.index_path):
    for index, train_inputs in enumerate(tqdm(instance_train_data_loader)):
        grad_z = compute_gradients(
            n_gpu=1,
            device=torch.device("cuda:0"),
            model=model,
            inputs=train_inputs,
            params_filter=params_filter,
            weight_decay=0.0,
            weight_decay_ignores=weight_decay_ignores,
        )
        # flatten
        grad_z = torch.cat([g.reshape(-1) for g in grad_z], axis=0).detach()
        accum_grads.append(grad_z.detach().cpu().to(torch.float32))
        if index % grad_batch == 0:
            # add to index
            grad_index.add(torch.stack(accum_grads).numpy())
            accum_grads = []
    faiss.write_index(grad_index, args.index_path)
    # del and reload so we can use mmap (save memory!)
    del grad_index

grad_index = faiss.read_index(args.index_path, faiss.IO_FLAG_MMAP)


# influence calculations. Since we are just doing gradient matching,
# we can just iterate and grab as we go.
instance_to_influences = {}
for index, instance in tqdm(enumerate(eval_data_loader), total=len(eval_data_loader)):
    x = 100
    grad_test = compute_gradients(
        n_gpu=1,
        device=torch.device("cuda:0"),
        model=model,
        inputs=instance,
        params_filter=params_filter,
        weight_decay=0.0,
        weight_decay_ignores=weight_decay_ignores,
    )
    grad_test = torch.cat([g.reshape(-1) for g in grad_test], axis=0).detach().cpu().to(torch.float32)
    grad_test = grad_test.reshape(1, -1)  # faiss expects a batch dimension
    # search for closest instance
    influences, topk_indices = grad_index.search(grad_test.numpy(), args.top_k)
    # save
    index_to_influence = {ind: influence for influence, ind in zip(influences[0], topk_indices[0])}
    instance_to_influences[index] = index_to_influence
    # periodically save to disk to avoid losing progress
    if index % 100 == 0:
        with open(args.instance_to_influences, "wb") as f:
            pickle.dump(instance_to_influences, f)
        print(f"Saved to {args.instance_to_influences} at step {index}")
