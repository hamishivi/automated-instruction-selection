"""
Doing the magnitude calculations similar to the 'text-to-text learner task conflict' paper
"""
import argparse
import gzip
import json
import os
from typing import Any, Dict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompts",
    type=str,
    help="Text file with a list of P3 prompt names (one per line) we want to analyze",
)
parser.add_argument(
    "--datasets",
    type=str,
    help="Text file with a list of P3 dataset prefixes to group the prompts by",
)
parser.add_argument("--p3_data", type=str, help="Gzipped P3 jsonl data file")
parser.add_argument(
    "--p3_indices",
    type=str,
    help="Gzipped file with dataset names corresponding to the P3 data file",
)
parser.add_argument("--max_instances_per_dataset", type=int, default=1000)
parser.add_argument("--model", type=str, default="google/t5-xl-lm-adapt")
parser.add_argument(
    "--encoder_block_name",
    type=str,
    default="encoder.block.23.layer.1",
    help="Default corresponds to FF in the final Transformer layer",
)
parser.add_argument("--run_pca", action="store_true")
parser.add_argument("--num_pca_components", type=int, default=100)
parser.add_argument(
    "--computed_distances", type=str, help="Pickle file to store computed pairwise distances"
)
parser.add_argument("--mag_output", type=str, help="Pickle file to store computed norms")
args = parser.parse_args()


prompts_of_interest = [x.strip() for x in open(args.prompts)]
datasets = [x.strip() for x in open(args.datasets)]
prompt_2_dataset = {}
for prompt in prompts_of_interest:
    for ds in datasets:
        if ds.lower() in prompt.lower():
            prompt_2_dataset[prompt] = ds

if args.computed_distances is None or not os.path.exists(args.computed_distances):
    print("Reading P3 data")
    p3_data_filename = args.p3_data
    p3_dataset_indices_filename = args.p3_indices
    text_data: Dict[str, Any] = {x: [] for x in prompts_of_interest}
    p3_data_ptr = gzip.open(p3_data_filename, "rt")
    p3_indices_ptr = gzip.open(p3_dataset_indices_filename, "rt")
    for data_line, dataset_indices_line in tqdm(zip(p3_data_ptr, p3_indices_ptr)):
        dataset_name = dataset_indices_line.split("\t")[0]
        if dataset_name not in text_data:
            continue
        if len(text_data[dataset_name]) >= args.max_instances_per_dataset:
            continue
        text_data[dataset_name].append(json.loads(data_line))
        if all([len(text_data[x]) >= args.max_instances_per_dataset for x in text_data]):
            break

    print("Read instances from datasets:")
    for dataset_name in prompts_of_interest:
        print(f"\t{dataset_name}\t{len(text_data[dataset_name])}")
    p3_data_ptr.close()
    p3_indices_ptr.close()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.cuda()
    parameters_of_interest = []
    for name, parameter in model.named_parameters():
        if name.startswith("encoder.final_layer_norm"):
            # if 'layer_norm' in name.lower() or 'layernorm' in name.lower():
            parameters_of_interest.append((name, parameter))

    print(f"Computing gradients on {args.model}")
    # print(f"Computing gradients only on {args.encoder_block_name} and the final layer norm weight")

    all_task_gradients: Dict[str, Any] = {}
    for dataset_name in tqdm(prompts_of_interest):
        dataset_prefix = prompt_2_dataset[dataset_name]
        instances = text_data[dataset_name]
        all_task_gradients[dataset_prefix] = None
        for instance in tqdm(instances):
            inputs = tokenizer.encode(
                instance["input"], return_tensors="pt", truncation=True
            ).cuda()
            targets = tokenizer.encode(instance["target"], return_tensors="pt").cuda()
            model_outputs = model(input_ids=inputs, labels=targets, return_dict=True)
            loss = model_outputs["loss"]
            loss.backward(inputs=[p for n, p in parameters_of_interest])

            gradients = (
                torch.cat([p.grad.flatten() for _, p in parameters_of_interest])
                .detach()
                .cpu()
                .numpy()
            )
            if all_task_gradients[dataset_prefix] is None:
                all_task_gradients[dataset_prefix] = gradients
            else:
                all_task_gradients[dataset_prefix] += gradients
            model.zero_grad()
        # average out the task gradient
        all_task_gradients[dataset_prefix] /= args.max_instances_per_dataset
        # all_task_gradients[dataset_name] = np.stack(all_task_gradients[dataset_name], axis=0).mean(axis=0)

# compute the overall average
all_task_gradient = np.stack(all_task_gradients.values(), axis=0).mean(axis=0)

with open(args.mag_output, "w") as outfile:
    print("Average gradient norm:", np.linalg.norm(all_task_gradient), file=outfile)
    print("Per-task norms:", file=outfile)
    norms = []
    for dataset_prefix in datasets:
        if dataset_prefix in all_task_gradients:
            norms.append(np.linalg.norm(all_task_gradients[dataset_prefix]))
            print(f"{dataset_prefix}\t{norms[-1]}", file=outfile)
    # print("Overall variance (magnitude conflict):", np.var(norms), file=outfile)
