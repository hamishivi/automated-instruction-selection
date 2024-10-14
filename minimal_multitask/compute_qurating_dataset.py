'''
Calculate qurating scores and keep top x
'''
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import json
import numpy as np
import os

from minimal_multitask.utils import create_prompt_with_tulu_chat_format

parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset', type=str, help='input file')
parser.add_argument('--n', type=int, default=10000, help='top n qurating scores to keep')
parser.add_argument('--output_file', type=str, help='output file')
args = parser.parse_args()

model = AutoModelForSequenceClassification.from_pretrained("princeton-nlp/QuRater-1.3B", trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/QuRater-1.3B")

# load and process train dataset
if args.train_dataset == "alpaca":
    og_train_dataset = load_dataset("json", data_files="data/camel_datasets/stanford_alpaca/stanford_alpaca_data.jsonl")[
        "train"
    ]
    train_dataset = og_train_dataset.map(
        lambda x: create_prompt_with_tulu_chat_format(x['messages'], tokenizer), num_proc=16
    )
elif args.train_dataset == "tulu2":
    og_train_dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
    train_dataset = og_train_dataset.map(
        lambda x: {"text": create_prompt_with_tulu_chat_format(x['messages'], tokenizer)}, num_proc=16
    )
    train_dataset = train_dataset['text']
else:
    if os.path.exists(args.train_dataset):
        og_train_dataset = load_dataset("json", data_files=args.train_dataset)["train"]
        train_dataset = og_train_dataset.map(
            lambda x: {"text": create_prompt_with_tulu_chat_format(x['messages'], tokenizer)}, num_proc=16
        )
        train_dataset = train_dataset['text']
    else:
        raise ValueError(f"Invalid train dataset: {args.train_dataset}")
print(f"Train dataset size: {len(train_dataset)}")

# for each prompt, calculate qurating score
qurating_scores = []
for prompt in tqdm(train_dataset):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**{k: v.cuda() for k, v in inputs.items()})
    # should be 4 scores
    qurating_scores.append(outputs.logits[0].detach().cpu().numpy())

qurating_scores = np.array(qurating_scores)
# normalize qurating scores so each score is in [0, 1]
qurating_scores = (qurating_scores - qurating_scores.min(axis=0)) / (qurating_scores.max(axis=0) - qurating_scores.min(axis=0))
# take the average of the 4 scores
qurating_scores = qurating_scores.mean(axis=1)

print("min qurating score:", qurating_scores.min())
print("max qurating score:", qurating_scores.max())
# take top n
top_indices = np.argsort(qurating_scores)[-args.n:]
# go through original dataset and grab, and add score
top_prompts = []
for i in top_indices:
    instance = og_train_dataset[i.item()]
    instance['qurating'] = qurating_scores[i.item()].item()
    top_prompts.append(instance)
# save
with open(args.output_file, 'w') as f:
    for instance in top_prompts:
        f.write(json.dumps(instance) + '\n')
print(f"Saved top {args.n} qurating scores to {args.output_file}")
