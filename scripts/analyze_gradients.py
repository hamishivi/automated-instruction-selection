import gzip
import argparse
import os
import pickle
import json
import re
from tqdm import tqdm
import numpy
import torch
from sklearn.decomposition import PCA
from fastdist import fastdist
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str, help="Text file with a list of P3 dataset names (one per line) we want to analyze")
parser.add_argument("--p3_data", type=str, help="Gzipped P3 jsonl data file")
parser.add_argument("--p3_indices", type=str, help="Gzipped file with dataset names corresponding to the P3 data file")
parser.add_argument("--max_instances_per_dataset", type=int, default=1000)
parser.add_argument("--model", type=str, default="google/t5-xl-lm-adapt")
parser.add_argument(
    "--parameter_name_regex",
    type=str,
    nargs="+",
    default=["encoder\.block\.23\.layer\.1.*", "encoder\.final_layer_norm"],
    help="Multiple regexes to specify a set of parameters of interest"
)
parser.add_argument("--run_pca", action="store_true")
parser.add_argument("--num_pca_components", type=int, default=100)
parser.add_argument("--computed_gradients", type=str, help="Pickle file to store computed gradients")
parser.add_argument("--computed_distances", type=str, help="Pickle file to store computed pairwise distances")
parser.add_argument("--output", type=str, help="TSV file where intra and inter dataset distances will be written")
args = parser.parse_args()


datasets_of_interest = [x.strip() for x in open(args.datasets)]
if args.computed_distances is None or not os.path.exists(args.computed_distances):
    if args.computed_gradients is None or not os.path.exists(args.computed_gradients):
        print("Reading P3 data")
        p3_data_filename = args.p3_data
        p3_dataset_indices_filename = args.p3_indices
        text_data = {x: [] for x in datasets_of_interest}
        p3_data_ptr = gzip.open(p3_data_filename, "rt")
        p3_indices_ptr = gzip.open(p3_dataset_indices_filename, "rt")
        for data_line, dataset_indices_line in tqdm(zip(p3_data_ptr, p3_indices_ptr)):
            # P3 dataset names also have prompt template IDs concatenated to them. We want the logic to work even if the ``dataset_of_interest``
            # list does not have those prompt template IDs.
            dataset_name_with_prompt = dataset_indices_line.split("\t")[0]
            dataset_name = None
            for name in datasets_of_interest:
                if dataset_name_with_prompt.startswith(name):
                    dataset_name = name
                    break
            if dataset_name is None:
                continue
            if len(text_data[dataset_name]) >= args.max_instances_per_dataset:
                continue
            text_data[dataset_name].append(json.loads(data_line))
            if all([len(text_data[x]) >= args.max_instances_per_dataset for x in text_data]):
                break

        print("Read instances from datasets:")
        for dataset_name in datasets_of_interest:
            print(f"\t{dataset_name}\t{len(text_data[dataset_name])}")
        p3_data_ptr.close()
        p3_indices_ptr.close()


        tokenizer = AutoTokenizer.from_pretrained(args.model)
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
        for dataset_name in tqdm(datasets_of_interest):
            instances = text_data[dataset_name]
            for instance in tqdm(instances):
                inputs = tokenizer.encode(instance["input"], truncation=True, return_tensors="pt").cuda()
                targets = tokenizer.encode(instance["target"], truncation=True, return_tensors="pt").cuda()
                model_outputs = model(input_ids=inputs, labels=targets, return_dict=True)
                loss = model_outputs['loss']
                loss.backward(inputs=[p for _, p in parameters_of_interest])

                gradients = torch.cat([p.grad.flatten() for _, p in parameters_of_interest]).detach().cpu().numpy()
                all_dataset_gradients.append(gradients)
                model.zero_grad()
            
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


dataset_index_ranges = {
        name: (
            i * args.max_instances_per_dataset,
            (i + 1) * args.max_instances_per_dataset
            )
        for i, name in enumerate(datasets_of_interest)
        }

with open(args.output, "w") as outfile:
    print("Intra dataset averages", file=outfile)
    for dataset_name in datasets_of_interest:
        i, j = dataset_index_ranges[dataset_name]
        print(f"{dataset_name}\t{distances[i:j, i:j].mean()}", file=outfile)
        
    print("\nInter dataset averages", file=outfile)
    print("\t".join([""] + datasets_of_interest), file=outfile)
    for dataset_name1 in datasets_of_interest:
        d1_distances = []
        for dataset_name2 in datasets_of_interest:
            i1, j1 = dataset_index_ranges[dataset_name1]
            i2, j2 = dataset_index_ranges[dataset_name2]
            d1_distances.append(distances[i1:j1, i2:j2].mean())
        output_line = "\t".join([dataset_name1] + [str(d) for d in d1_distances])
        print(output_line, file=outfile)
