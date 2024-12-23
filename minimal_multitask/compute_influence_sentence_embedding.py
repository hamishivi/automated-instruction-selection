import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np

from minimal_multitask.data import DATASETS
from minimal_multitask.utils import create_prompt_with_tulu_chat_format

from tqdm import tqdm
import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="nvidia/NV-Embed-v2")
parser.add_argument("--no_query_prefix", action="store_true")
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
args = parser.parse_args()


torch.manual_seed(args.seed)
if args.dtype == "bf16":
    kwargs = {"torch_dtype": torch.bfloat16}
elif args.dtype == "fp16":
    kwargs = {"torch_dtype": torch.float16}
elif args.dtype == "fp32":
    kwargs = {"torch_dtype": torch.float32}
# kwargs["use_flash_attention_2"] = True

if args.model_name_or_path.startswith("sentence-transformers/"):
    model = SentenceTransformer(args.model_name_or_path, model_kwargs=kwargs)
else:
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        **kwargs,
    )
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)


def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


def remove_extra_assistant_tokens(sample):
    sample['text'] = rreplace(sample['text'], "<|assistant|>", "", 1).strip()
    return sample


# load and process train dataset
if args.train_dataset == "alpaca":
    train_dataset = load_dataset("json", data_files="data/stanford_alpaca_data.jsonl")[
        "train"
    ]
    train_dataset = train_dataset.map(
        lambda x: create_prompt_with_tulu_chat_format(x['messages'], tokenizer, prompt_only=args.prompt_only, response_only=args.label_only, no_special_tokens=True), num_proc=16
    )
    train_dataset = train_dataset.map(remove_extra_assistant_tokens, num_proc=16)
elif args.train_dataset == "tulu2":
    train_dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
    train_dataset = train_dataset.map(
        lambda x: {"text": create_prompt_with_tulu_chat_format(x['messages'], tokenizer, prompt_only=args.prompt_only, response_only=args.label_only, no_special_tokens=True)}, num_proc=16
    )
    train_dataset = train_dataset.map(remove_extra_assistant_tokens, num_proc=16)
    train_dataset = train_dataset['text']
else:
    if os.path.exists(args.train_dataset):
        train_dataset = load_dataset("json", data_files=args.train_dataset)["train"]
        train_dataset = train_dataset.map(
            lambda x: {"text": create_prompt_with_tulu_chat_format(x['messages'], tokenizer, prompt_only=args.prompt_only, response_only=args.label_only, no_special_tokens=True)}, num_proc=16, load_from_cache_file=False
        )
        train_dataset = train_dataset.map(remove_extra_assistant_tokens, num_proc=16)
        train_dataset = train_dataset['text']
    else:
        raise ValueError(f"Invalid train dataset: {args.train_dataset}")

# test dataset - mostly handled in data.py
if args.eval_dataset in DATASETS:
    test_dataset = DATASETS[args.eval_dataset](tokenizer).get_all_test_prompts(
        seed=args.seed, prompt_only=args.prompt_only, response_only=args.label_only
    )
    # gonna be annoying and just decode the test prompts to get text.
    test_dataset = test_dataset.map(
        lambda x: {"text": tokenizer.decode(x["input_ids"], skip_special_tokens=True)}, num_proc=16
    )
    test_dataset = test_dataset['text']
else:
    raise ValueError(f"Invalid dataset: {args.dataset}")

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# prefixes
passage_prefix = ""
query_prefix = "Instruct: Given a sample, find the passages closest to that sample.\nQuery: "

# construct dataloaders
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
eval_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

if args.index_path is not None and os.path.exists(args.index_path):
    all_train_embeds = torch.load(args.index_path)
else:
    all_train_embeds = []
    for index, train_inputs in enumerate(tqdm(train_data_loader)):
        with torch.no_grad():
            max_length = 8192
            passage_embeddings = model.encode(train_inputs, instruction=passage_prefix, max_length=max_length)
            if isinstance(passage_embeddings, np.ndarray):
                passage_embeddings = torch.from_numpy(passage_embeddings).cuda()
            passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)

        all_train_embeds.append(passage_embeddings.detach().cpu())
        torch.cuda.empty_cache()

    all_train_embeds = torch.cat(all_train_embeds, dim=0)
    with open(args.index_path, "wb") as f:
        torch.save(all_train_embeds, f)

sim_influences = []
for idx, test_inputs in enumerate(tqdm(eval_data_loader)):
    with torch.no_grad():
        max_length = 8192
        query_embeddings = model.encode(test_inputs, instruction=query_prefix, max_length=max_length)
        if isinstance(query_embeddings, np.ndarray):
            query_embeddings = torch.from_numpy(query_embeddings).cuda()
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    influences = (query_embeddings @ all_train_embeds.T.cuda()).detach().cpu()
    sim_influences.append(influences)

sim_influences = torch.cat(sim_influences, dim=0)

# Convert to dictionary format
influence_dict = {}
for i in range(sim_influences.shape[0]):
    influence_dict[i] = {j: sim_influences[i][j].item() for j in range(sim_influences.shape[1])}

# save the influences
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
with open(
    os.path.join(args.save_dir, f"{args.eval_dataset}_embedding.pkl"),
    "wb",
) as f:
    pickle.dump(influence_dict, f)
print('saved', os.path.join(args.save_dir, f"{args.eval_dataset}_embedding.pkl"))
