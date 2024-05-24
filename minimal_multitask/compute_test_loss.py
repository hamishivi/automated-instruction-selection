'''
Script for computing the test loss for a given dataset with a given model.
We can compute loss over dev splits or given prompt files.
'''
import argparse
import json
import os

from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.create_llama_encodings import encode_with_messages_format
from minimal_multitask.data import DATASETS


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf')
parser.add_argument('--tokenizer', type=str, default=None)
parser.add_argument('--eval_datasets', nargs='+', default=['mmlu'])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--results', type=str, default=None)
args = parser.parse_args()

# load model
kwargs = dict(device_map='auto', use_auth_token=os.environ.get('HF_TOKEN', None))
model = AutoModelForCausalLM.from_pretrained(args.model_name, )
tokenizer = AutoTokenizer.from_pretrained(args.model_name if args.tokenizer is None else args.tokenizer)

results = {}
# big loop over all eval datasets
for dataset in args.eval_datasets:
    # load and process eval dataset
    if dataset in DATASETS:
        test_dataset = DATASETS[dataset](tokenizer).get_all_test_prompts()
        eval_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    else:
        # assume its a jsonl file in the messages format
        data = [json.loads(line) for line in open(dataset, 'r')]
        # convert to prompt format
        train_dataset = [encode_with_messages_format(x, tokenizer, 2048, True, False) for x in data]
        eval_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    # for each prompt, compute loss
    # todo: maybe make this batched? But then is not properly per-sample avg.
    losses = []
    with torch.inference_mode():
        for index, train_inputs in enumerate(tqdm(eval_data_loader)):
            loss = model(**{k: v.to('cuda') for k,v in train_inputs.items()}).loss
            losses.append(loss.item())
    print(f"Average loss for {dataset}: {sum(losses) / len(losses)}")
    results[dataset] = {"Average loss": sum(losses) / len(losses)}
# output to metrics file if given (for nicer beaker integration)
if args.results:
    with open(args.results, 'w') as f:
        f.write(json.dumps(results, indent=4))
