'''
Compute influence using logix package.
Wip.
'''
import os
import argparse
import copy
import pickle
import yaml

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import logix
from logix.utils import merge_logs
from accelerate import Accelerator
from tqdm import tqdm

from scripts.create_llama_encodings import encode_with_messages_format
from minimal_multitask.data import DATASETS


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf')
parser.add_argument('--tokenizer_name', type=str, default=None)
parser.add_argument('--train_dataset', type=str, default='tulu2')
parser.add_argument('--gradient_checkpointing', action='store_true')
parser.add_argument('--use_flash_attention_2', action='store_true')
parser.add_argument('--use_hf_auth_token', type=str, default=None)
parser.add_argument('--eval_dataset', type=str, default='gsm8k_shots')
parser.add_argument('--instance_to_influences', type=str, required=True)
parser.add_argument('--grad_save_path', type=str, default=None)  # if we have saved grads, we can use them
args = parser.parse_args()

accelerator = Accelerator()

kwargs = {}
if args.use_flash_attention_2:
    kwargs["use_flash_attention_2"] = True
if args.use_hf_auth_token is not None:
    kwargs['use_auth_token'] = os.environ.get('HF_TOKEN', None)

# load model
model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, **kwargs)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.model_name)

if args.gradient_checkpointing:
    model.gradient_checkpointing_enable()

# load and process train dataset
# TODO: fix openorca system message bug
if args.train_dataset == 'alpaca':
    train_dataset = load_dataset('json', data_files='data/camel_datasets/stanford_alpaca/stanford_alpaca_data.jsonl')['train']
    train_dataset = train_dataset.map(lambda x: encode_with_messages_format(x, tokenizer, 512, True, False), num_proc=16)
elif args.train_dataset == 'tulu2':
    train_dataset = load_dataset('allenai/tulu-v2-sft-mixture', split='train')
    train_dataset = train_dataset.map(lambda x: encode_with_messages_format(x, tokenizer, 512, True, False), num_proc=16)
else:
    if os.path.exists(args.train_dataset):
        train_dataset = load_dataset('json', data_files=args.train_dataset)['train']
        train_dataset = train_dataset.map(lambda x: encode_with_messages_format(x, tokenizer, 2048, True, False), num_proc=16)
    else:
        raise ValueError(f"Invalid train dataset: {args.train_dataset}")
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

data_loader = DataLoader(train_dataset, batch_size=1)

name_filter = ["att", "mlp"]

# construct the logix yaml here.
# annoyingly, they dont allow non-yaml construction rn, so we just make a yaml on the fly
logix_config = {
    "root_dir": ".",
    "logging": {
        "flush_threshold": 1000000000,
        "num_workers": 2,
        "cpu_offload": True,
        "log_dtype": "float16",
    },
    "lora": {
        "init": "random",
        "rank": 6,
    }
}
os.makedirs("tmp_logix", exist_ok=True)
with open("tmp_logix/logix_config.yaml", "w") as f:
    yaml.dump(logix_config, f)

if args.grad_save_path is None or not os.path.exists(args.grad_save_path):
    # we need to index train data, lets go ahead and do that
    # logix setup
    run = logix.init(project=args.grad_save_path, config='tmp_logix/logix_config.yaml')

    # Specify modules to be tracked for logging
    run.watch(model, name_filter=name_filter, type_filter=[nn.Linear])
    # build scheduler
    scheduler = logix.LogIXScheduler(
        run, lora="random", hessian="none", save="grad"
    )

    # compute influences
    model, data_loader = accelerator.prepare(model, data_loader)
    model.eval()
    for _ in scheduler:
        for batch in tqdm(data_loader):
            data_id = tokenizer.batch_decode(batch["input_ids"])
            targets = batch.pop("labels")
            with run(data_id=data_id, mask=batch["attention_mask"]):
                model.zero_grad()
                lm_logits = model(**batch).logits
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = targets[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="sum",
                    ignore_index=-100,
                )
                accelerator.backward(loss)
        logix.finalize()
else:
    # we can just initialize the logix run from the saved grads
    run = logix.init(args.grad_save_path, config='tmp_logix/logix_config.yaml')
    logix.watch(model, name_filter=name_filter)
    logix.initialize_from_log()
    # extra setup
    model = accelerator.prepare(model)
    model.eval()

# setup log dataloader (dataloader over train grads)
log_loader = logix.build_log_dataloader(batch_size=64)

# now, we compute influence scores
# test dataset - mostly handled in data.py
if args.eval_dataset in DATASETS:
    test_dataset = DATASETS[args.eval_dataset](tokenizer).get_all_test_prompts()
else:
    raise ValueError(f"Invalid dataset: {args.dataset}")
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_data_loader = DataLoader(test_dataset, batch_size=1)
test_data_loader = accelerator.prepare(test_data_loader)

logix.setup({"grad": ["log"]})
logix.eval()
merged_test_logs = []
for idx, batch in enumerate(tqdm(test_data_loader)):
    data_id = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
    targets = batch.pop("labels")
    with run(data_id=data_id, mask=batch["attention_mask"]):
        model.zero_grad()
        logits = model(**batch).logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
            ignore_index=-100,
        )
        accelerator.backward(loss)

    test_log = logix.get_log()
    merged_test_logs.append(copy.deepcopy(test_log))

merged_test_log = merge_logs(merged_test_logs)
results = run.influence.compute_influence_all(merged_test_log, log_loader)

influence_scores = results["influence"]
# invert the influence score so lower is better, as with our other influences.
# TODO: double check inverting is the right move here.
influence_scores = -influence_scores

# construct the influence to index pickle
influence_to_index = []
for test_index, influence in enumerate(influence_scores):
    influence_to_index.append(
        {ind: float(influence.item()) for ind, influence in enumerate(influence)}
    )
# save the influence to index pickle
with open(args.instance_to_influences, "wb") as f:
    pickle.dump(influence_to_index, f)