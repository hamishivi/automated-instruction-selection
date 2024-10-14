import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from minimal_multitask.data import DATASETS
from minimal_multitask.utils import encode_with_messages_format

from tqdm import tqdm
import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="huggyllama/llama-7b")
parser.add_argument("--tokenizer", type=str, default=None)
parser.add_argument("--save_dir", type=str, default="l")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--train_dataset", type=str, default="alpaca")
parser.add_argument("--eval_dataset", type=str, choices=DATASETS.keys(), default="mmlu")
parser.add_argument("--index_path", type=str)
# be careful with this one! leaks test data into train set so we can sanity check the retrieval
parser.add_argument("--leak_test_data", action="store_true")
parser.add_argument("--dtype", default="bf16")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--prompt_only", action="store_true")
parser.add_argument("--label_only", action="store_true")
parser.add_argument("--pooling", type=str, default="none") # none, mean, weighted_mean
parser.add_argument("--only_first_two", action="store_true")  # only use the first two messages
args = parser.parse_args()

assert args.pooling in ["none", "mean", "weighted_mean"]

torch.manual_seed(args.seed)
if args.dtype == "bf16":
    kwargs = {"torch_dtype": torch.bfloat16}
elif args.dtype == "fp16":
    kwargs = {"torch_dtype": torch.float16}
elif args.dtype == "fp32":
    kwargs = {"torch_dtype": torch.float32}
if "llama" in args.model_name:
    kwargs["use_flash_attention_2"] = True

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
            lambda x: encode_with_messages_format(x, tokenizer, 2048, True, args.label_only, args.only_first_two, args.prompt_only), num_proc=16, load_from_cache_file=False
        )
    else:
        raise ValueError(f"Invalid train dataset: {args.train_dataset}")
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# test dataset - mostly handled in data.py
if args.eval_dataset in DATASETS:
    test_dataset = DATASETS[args.eval_dataset](tokenizer).get_all_test_prompts(
        seed=args.seed, prompt_only=args.prompt_only, response_only=args.label_only
    )
else:
    raise ValueError(f"Invalid dataset: {args.dataset}")

if args.leak_test_data:
    # shrink the training data for quicker testing
    train_dataset = train_dataset.select(range(len(test_dataset)))
    # add test data to train data
    for sample in test_dataset:
        train_dataset = train_dataset.add_item({k: v.tolist() for k, v in sample.items()})

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# construct dataloaders
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
eval_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
if args.index_path is not None and os.path.exists(args.index_path):
        all_train_embeds = torch.load(args.index_path)
else:
    all_train_embeds = []
    for index, train_inputs in enumerate(tqdm(train_data_loader)):
        # if index > 10:
        #     break
        with torch.no_grad():
            train_outputs = model(
                **{k: v.to(model.device) for k, v in train_inputs.items() if k != "labels"}, output_hidden_states=True
            )
        label_len = torch.sum(train_inputs["labels"] != -100, dim=1)
        input_lens = torch.sum(train_inputs["attention_mask"], dim=1)
        # Get the mean hidden state corresponding to the label
        if args.pooling == "mean":
            train_embeddings = torch.mean(train_outputs["hidden_states"][-1], dim=1)
            train_embeddings = train_embeddings.unsqueeze(1)  # for compat
        elif args.pooling == "weighted_mean":
            # for this, we follow SGPT idea: https://arxiv.org/abs/2202.08904
            hidden_states = train_outputs["hidden_states"][-1]
            weighting_mask = torch.arange(hidden_states.size(1), device=hidden_states.device).unsqueeze(0) + 1  # +1 
            weighting_mask = weighting_mask / weighting_mask.sum(dim=1, keepdim=True)
            train_embeddings = torch.sum(hidden_states * weighting_mask.unsqueeze(-1), dim=1)
            train_embeddings = train_embeddings.unsqueeze(1)  # for compat
        else:
            # just use the last token hiden state
            train_embeddings = train_outputs["hidden_states"][-1][:, input_lens - 1]
        all_train_embeds.append(train_embeddings[:,0])

    all_train_embeds = torch.cat(all_train_embeds, dim=0)
    all_train_embeds = all_train_embeds / torch.linalg.vector_norm(all_train_embeds, dim=1, keepdim=True)
    with open(args.index_path, "wb") as f:
        torch.save(all_train_embeds, f)

sim_influences = []
for idx, test_inputs in enumerate(tqdm(eval_data_loader)):
    with torch.no_grad():
        test_outputs = model(
            **{k: v.to(model.device) for k, v in test_inputs.items() if k != "labels"}, output_hidden_states=True
        )
    label_len = torch.sum(test_inputs["labels"] != -100, dim=1)
    input_lens = torch.sum(test_inputs["attention_mask"], dim=1)
    # Get the mean hidden state corresponding to the label
    if args.pooling == "mean":
        test_embeddings = torch.mean(test_outputs["hidden_states"][-1], dim=1)
        test_embeddings = test_embeddings.unsqueeze(1)  # for compat
    elif args.pooling == "weighted_mean":
        # for this, we follow SGPT idea: https://arxiv.org/abs/2202.08904
        hidden_states = test_outputs["hidden_states"][-1]
        weighting_mask = torch.arange(hidden_states.size(1), device=hidden_states.device).unsqueeze(0) + 1  # +1 
        weighting_mask = weighting_mask / weighting_mask.sum(dim=1, keepdim=True)
        test_embeddings = torch.sum(hidden_states * weighting_mask.unsqueeze(-1), dim=1)
        test_embeddings = test_embeddings.unsqueeze(1)  # for compat
    else:
        # just use the last token hiden state
        test_embeddings = test_outputs["hidden_states"][-1][:, input_lens - 1]

    test_embeddings = test_embeddings.squeeze(1)
    test_embeddings = test_embeddings / torch.linalg.vector_norm(test_embeddings, dim=1, keepdim=True)
    all_train_embeds = all_train_embeds.squeeze()
    influences = torch.matmul(
        test_embeddings / torch.linalg.vector_norm(test_embeddings, dim=1, keepdim=True), all_train_embeds.T
    )
    sim_influences.append(influences)

sim_influences = torch.cat(sim_influences, dim=0)

# Convert to dictionary format
influence_dict = {}
for i in range(sim_influences.shape[0]):
    influence_dict[i] = {j: sim_influences[i][j].item() for j in range(sim_influences.shape[1])}

# save the influences
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if args.prompt_only:
    with open(
        os.path.join(
            args.save_dir, f"{args.eval_dataset}_cossim_promptonly.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(influence_dict, f)
elif args.label_only:
    with open(
        os.path.join(
            args.save_dir, f"{args.eval_dataset}_cossim_labelonly.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(influence_dict, f)
else:
    with open(
        os.path.join(args.save_dir, f"{args.eval_dataset}_cossim.pkl"),
        "wb",
    ) as f:
        pickle.dump(influence_dict, f)
    print('saved', os.path.join(args.save_dir, f"{args.eval_dataset}_cossim.pkl"))