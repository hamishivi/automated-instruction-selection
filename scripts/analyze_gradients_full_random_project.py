import argparse
import gzip
import json
import os
import pickle
import re
from typing import Any, Dict

import numpy
import torch
from fastdist import fastdist
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from minimal_multitask.dataset_mapping import PROMPT_MAPPING

parser = argparse.ArgumentParser()
parser.add_argument(
    "--datasets",
    type=str,
    help="Text file with a list of P3 dataset names we want to analyze, optionally with a tab-separated mapping",
)
parser.add_argument(
    "--split_name",
    type=str,
    default="train",
    help="The split in each dataset to use data from (default is 'train')",
)
parser.add_argument("--p3_data", type=str, help="Gzipped P3 jsonl data file")
parser.add_argument(
    "--p3_indices",
    type=str,
    help="Gzipped file with dataset names corresponding to the P3 data file",
)
parser.add_argument("--max_instances_per_dataset", type=int, default=1000)
parser.add_argument("--model", type=str, default="google/t5-xl-lm-adapt")
parser.add_argument("--tokenizer", type=str, default="google/t5-xl-lm-adapt")
parser.add_argument(
    "--parameter_name_regex",
    type=str,
    nargs="+",
    default=[r"encoder\.block\.23\.layer\.1.*", r"encoder\.final_layer_norm"],
    help="Multiple regexes to specify a set of parameters of interest",
)
parser.add_argument(
    "--random_weights",
    action="store_true",
    help="Randomly initialize model instead of downloading pretrained weights",
)
parser.add_argument("--run_pca", action="store_true")
parser.add_argument("--num_pca_components", type=int, default=100)
parser.add_argument("--computed_gradients", type=str, help="Pickle file to store computed gradients")
parser.add_argument("--computed_distances", type=str, help="Pickle file to store computed pairwise distances")
parser.add_argument("--print_intra_dataset_distances", action="store_true")
parser.add_argument("--output", type=str, help="TSV file where intra and inter dataset distances will be written")
parser.add_argument(
    "--project_gradient",
    action="store_true",
    help="Randomly project the gradient. Preserves distances to some degree. Required for large models.",
)
parser.add_argument(
    "--large_gradient_matmul",
    action="store_true",
    help="Use a batch matmul script. Required for gradients over max numpy int size (~2 billion)",
)
args = parser.parse_args()

numpy.random.seed(10)

dataset_name_lines = [x.strip() for x in open(args.datasets)]
if "\t" in dataset_name_lines[0]:
    # We have a mapping
    dataset_name_mapping = {}
    mapped_dataset_names = []
    datasets_of_interest = []
    for string in dataset_name_lines:
        name, mapping = string.split("\t")
        if mapping not in mapped_dataset_names:
            mapped_dataset_names.append(mapping)
        datasets_of_interest.append(name)
        dataset_name_mapping[name] = mapping
else:
    dataset_name_mapping = {x: x for x in dataset_name_lines}
    mapped_dataset_names = dataset_name_lines
    datasets_of_interest = dataset_name_lines

if args.computed_distances is None or not os.path.exists(args.computed_distances):
    if args.computed_gradients is None or not os.path.exists(args.computed_gradients):
        print("Reading P3 data")
        p3_data_filename = args.p3_data
        p3_dataset_indices_filename = args.p3_indices
        text_data: Dict[Any, Any] = {x: [] for x in mapped_dataset_names}
        p3_data_ptr = gzip.open(p3_data_filename, "rt")
        p3_indices_ptr = gzip.open(p3_dataset_indices_filename, "rt")
        for data_line, dataset_indices_line in tqdm(zip(p3_data_ptr, p3_indices_ptr)):
            data_line_parts = dataset_indices_line.split("\t")
            if data_line_parts[1] != args.split_name:
                continue
            dataset_name_with_prompt = data_line_parts[0]
            if dataset_name_with_prompt not in datasets_of_interest:
                continue
            mapped_dataset_name = dataset_name_mapping[dataset_name_with_prompt]
            if len(text_data[mapped_dataset_name]) >= args.max_instances_per_dataset:
                continue
            text_data[mapped_dataset_name].append(json.loads(data_line))
            if all([len(text_data[x]) >= args.max_instances_per_dataset for x in text_data]):
                break

        print("Read instances from datasets:")
        for mapped_dataset_name in mapped_dataset_names:
            print(f"\t{mapped_dataset_name}\t{len(text_data[mapped_dataset_name])}")
        p3_data_ptr.close()
        p3_indices_ptr.close()

        # re-organise data by task grouping prompts together.
        new_mapping: Dict[Any, Any] = {}
        new_text_data: Dict[Any, Any] = {}
        for mapped_dataset_name in mapped_dataset_names:
            task = PROMPT_MAPPING[mapped_dataset_name][0]
            if task not in new_mapping:
                new_mapping[task] = []
            new_mapping[task].append(mapped_dataset_name)
            if task not in new_text_data:
                new_text_data[task] = []
            new_text_data[task].extend(text_data[mapped_dataset_name])
        mapped_dataset_names = list(new_mapping.keys())
        text_data = new_text_data

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if args.random_weights:
            config = AutoConfig.from_pretrained(args.model)
            model = AutoModelForSeq2SeqLM.from_config(config)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        model.cuda()
        parameters_of_interest = []
        num_parameters = 0
        for name, parameter in model.named_parameters():
            if any([re.match(pattern, name) is not None for pattern in args.parameter_name_regex]):
                parameters_of_interest.append((name, parameter))
                num_parameters += parameter.numel()

        print(f"Computing gradients on {args.model}")
        print(f"Computing gradients only on {[x[0] for x in parameters_of_interest]}")
        print(f"\tThat's a total of {num_parameters} parameters")

        all_dataset_gradients = []
        dataset_index_ranges = {}
        index_counter = 0
        for mapped_dataset_name in tqdm(mapped_dataset_names, position=0):
            instances = text_data[mapped_dataset_name]
            dataset_index_ranges[mapped_dataset_name] = [index_counter, None]
            for instance in tqdm(instances, position=1):
                inputs = tokenizer.encode(instance["input"], truncation=True, return_tensors="pt").cuda()
                targets = tokenizer.encode(instance["target"], truncation=True, return_tensors="pt").cuda()
                model_outputs = model(input_ids=inputs, labels=targets, return_dict=True)
                loss = model_outputs["loss"]
                loss.backward(inputs=[p for _, p in parameters_of_interest])

                gradients = torch.cat([p.grad.flatten() for _, p in parameters_of_interest]).detach()
                model.zero_grad()
                # project gradient down
                gen = torch.Generator(device="cuda")
                gen.manual_seed(1000)
                with torch.inference_mode():
                    if args.project_gradient:
                        if args.large_gradient_matmul:
                            projected_gradients = []
                            for _ in tqdm(range(100), position=2, leave=False):
                                # rand_proj = torch.ones((gradients.shape[0], 1), device='cuda')
                                rand_proj = torch.normal(
                                    mean=0.0,
                                    std=1.0,
                                    size=(gradients.shape[0], 1),
                                    generator=gen,
                                    device="cuda",
                                )
                                # matmul has a max size... 😱 so we do it in chunks
                                # i tried to use int_max, but it gave weird cuda memory errors...
                                max_size = 2000000000
                                if rand_proj.shape[0] >= max_size:
                                    result_projs = []
                                    for i in range(0, gradients.shape[0], max_size):
                                        end = min(i + max_size, gradients.shape[0])
                                        result_projs.append(gradients[i:end] @ rand_proj[i:end])
                                    projected_gradients.append(torch.stack(result_projs, axis=0).sum(axis=0))
                                else:
                                    projected_gradients.append(gradients @ rand_proj)
                            gradients = torch.cat(projected_gradients, axis=0).cpu().numpy()
                        else:
                            rand_proj = torch.normal(
                                mean=0.0,
                                std=1.0,
                                size=(gradients.shape[0], 100),
                                generator=gen,
                                device="cuda",
                            )
                            gradients = (gradients @ rand_proj).detach().cpu().numpy()
                all_dataset_gradients.append(gradients)
                model.zero_grad()
            dataset_index_ranges[mapped_dataset_name][1] = index_counter + len(instances)
            index_counter += len(instances)

        all_gradients = numpy.stack(all_dataset_gradients)
        print(f"Done computing gradients. The size of the matrix is {all_gradients.shape}")

        if args.run_pca:
            print("Running PCA")
            pca = PCA(n_components=args.num_pca_components, random_state=0)
            all_gradients = pca.fit_transform(all_gradients)
            print(f"Done running PCA. The size of the gradients matrix is {all_gradients.shape}")

        if args.computed_gradients is not None:
            with open(args.computed_gradients, "wb") as outfile:
                pickle.dump(all_gradients, outfile)
    else:
        print(f"Loading gradients from {args.computed_gradients}")
        with open(args.computed_gradients, "rb") as infile:
            all_gradients = pickle.load(infile)

    print("Computing cosine distances")
    distances = 1 - fastdist.cosine_pairwise_distance(all_gradients, return_matrix=True)
    if args.computed_distances is not None:
        with open(args.computed_distances, "wb") as outfile:
            pickle.dump(distances, outfile)
else:
    print(f"Loading pairwise distances from {args.computed_distances}")
    with open(args.computed_distances, "rb") as infile:
        distances = pickle.load(infile)


# dataset_index_ranges = {
#     name: (i * args.max_instances_per_dataset, (i + 1) * args.max_instances_per_dataset)
#     for i, name in enumerate(mapped_dataset_names)
# }
mapped_dataset_names = sorted(mapped_dataset_names)
with open(args.output, "w") as outfile:
    print("\nInter dataset averages", file=outfile)
    print("\t".join([""] + mapped_dataset_names), file=outfile)
    for dataset_name1 in mapped_dataset_names:
        d1_distances = []
        for dataset_name2 in mapped_dataset_names:
            i1, j1 = dataset_index_ranges[dataset_name1]
            i2, j2 = dataset_index_ranges[dataset_name2]
            d1_distances.append(distances[i1:j1, i2:j2].mean())
        output_line = "\t".join([dataset_name1] + [str(d) for d in d1_distances])
        print(output_line, file=outfile)
