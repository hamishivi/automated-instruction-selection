import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from datasets import load_dataset, Dataset

from minimal_multitask.data import DATASETS
from minimal_multitask.utils import encode_with_messages_format, create_prompt_with_tulu_chat_format

from tqdm import tqdm
import argparse
import os
import pickle
import string

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
parser.add_argument("--use_ccds_processing", action="store_true")
args = parser.parse_args()


torch.manual_seed(args.seed)
if args.dtype == "bf16":
    kwargs = {"torch_dtype": torch.bfloat16}
elif args.dtype == "fp16":
    kwargs = {"torch_dtype": torch.float16}
elif args.dtype == "fp32":
    kwargs = {"torch_dtype": torch.float32}
# kwargs["use_flash_attention_2"] = True

model = AutoModel.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=True,
    **kwargs,
).cuda()
model = model.encoder
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)


def CCDS_processing_train(samples, tokenizer):
    messages = samples['messages']
    texts = [msg[0]['content'] for msg in messages]
    processed_texts = []
    for text in texts:
        text = text.lower()
        text = text.replace("\n", " ")
        # Get rid of chat formattting
        text = text.translate(str.maketrans('', '', string.punctuation))
        processed_texts.append(text)
    batch_dict = tokenizer(processed_texts, padding=True, max_length=256, truncation=True, return_tensors="np")

    return batch_dict


def CCDS_processing_test(samples, tokenizer):
    texts = tokenizer.batch_decode(samples['input_ids'], skip_special_tokens=True)
    processed_texts = []
    for text in texts:
        text = text.lower()
        text = text.replace("\n", " ")
        # Get rid of chat formattting
        text = text.replace("<|user|>", "")
        text = text.replace("<|assistant|>", "")
        # removing punctuation, why are we doing this?
        text = text.translate(str.maketrans('', '', string.punctuation))
        processed_texts.append(text) 
    batch_dict = tokenizer(processed_texts, padding=True, max_length=256, truncation=True, return_tensors="np")

    return batch_dict


# load and process train dataset
if args.train_dataset == "alpaca":
    train_dataset = load_dataset("json", data_files="data/stanford_alpaca_data.jsonl")[
        "train"
    ]
    train_dataset = train_dataset.map(
        lambda x: CCDS_processing_train(x, tokenizer), num_proc=16, batched=True, batch_size=16, load_from_cache_file=False
    )
elif args.train_dataset == "tulu2":
    train_dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
    train_dataset = train_dataset.map(
        lambda x: CCDS_processing_train(x, tokenizer), num_proc=16, batched=True, batch_size=16, load_from_cache_file=False
    )
else:
    if os.path.exists(args.train_dataset):
        train_dataset = load_dataset("json", data_files=args.train_dataset)["train"]
        train_dataset = train_dataset.map(
            lambda x: CCDS_processing_train(x, tokenizer), num_proc=16, batched=True, batch_size=16, load_from_cache_file=False
        )
    else:
        raise ValueError(f"Invalid train dataset: {args.train_dataset}")
train_dataset = train_dataset.remove_columns(['dataset', 'id', 'messages'])

# test dataset - mostly handled in data.py
if args.eval_dataset in DATASETS:
    test_dataset = DATASETS[args.eval_dataset](tokenizer).get_all_test_prompts(
        seed=args.seed, prompt_only=True
    )
    
    # gonna be annoying and just decode the test prompts to get text.
    test_dataset = test_dataset.map(
        lambda x: CCDS_processing_test(x, tokenizer), num_proc=16, batched=True, batch_size=16
    )
else:
    raise ValueError(f"Invalid dataset: {args.dataset}")
test_dataset = Dataset.from_dict({"input_ids": test_dataset['input_ids'], "attention_mask": test_dataset['attention_mask']})

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# prefixes
passage_prefix = ""
if not args.no_query_prefix:
    query_prefix = "Instruct: Given a sample, find the passages closest to that sample.\nQuery: "
else:
    query_prefix = ""

# construct dataloaders
data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)
eval_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

def mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    B, _S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs

if args.index_path is not None and os.path.exists(args.index_path):
    all_train_embeds = torch.load(args.index_path)
else:
    all_train_embeds = []
    for index, train_inputs in enumerate(tqdm(train_data_loader)):
        with torch.no_grad():
            passage_embeddings = model(input_ids=train_inputs['input_ids'].cuda(), attention_mask=train_inputs['attention_mask'].cuda())
            if hasattr(passage_embeddings, 'pooler_output'):
                passage_embeddings = passage_embeddings.pooler_output
            elif hasattr(passage_embeddings, 'last_hidden_state'):
                passage_embeddings = mean_pool(
                    hidden_states=passage_embeddings.last_hidden_state,
                    attention_mask=train_inputs['attention_mask'].cuda()
                )
            passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)

        all_train_embeds.append(passage_embeddings.detach().cpu())
        torch.cuda.empty_cache()

    all_train_embeds = torch.cat(all_train_embeds, dim=0)
    with open(args.index_path, "wb") as f:
        torch.save(all_train_embeds, f)

sim_influences = []
for idx, test_inputs in enumerate(tqdm(eval_data_loader)):
    with torch.no_grad():
        query_embeddings = model(input_ids=test_inputs['input_ids'].cuda(), attention_mask=test_inputs['attention_mask'].cuda())
        if hasattr(query_embeddings, 'pooler_output'):
            query_embeddings = outputs.pooler_output
        elif hasattr(query_embeddings, 'last_hidden_state'):
            query_embeddings = mean_pool(
                hidden_states=query_embeddings.last_hidden_state,
                attention_mask=test_inputs['attention_mask'].cuda()
            )
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
