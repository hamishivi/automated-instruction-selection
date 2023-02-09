import gzip
import argparse
import os
import pickle
import json
from tqdm import tqdm
import numpy
import torch
from sklearn.decomposition import PCA
from fastdist import fastdist
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str, help="Text file with a list of P3 dataset names (one per line) we want to analyze")
parser.add_argument("--p3_data", type=str, help="Gzipped P3 jsonl data file")
parser.add_argument("--p3_indices", type=str, help="Gzipped file with dataset names corresponding to the P3 data file")
parser.add_argument("--max_instances_per_dataset", type=int, default=1000)
parser.add_argument("--model", type=str, default="google/t5-xl-lm-adapt")
parser.add_argument("--encoder_block_name", type=str, default="encoder.block.23.layer.1", help="Default corresponds to FF in the final Transformer layer")
parser.add_argument("--run_pca", action="store_true")
parser.add_argument("--num_pca_components", type=int, default=100)
parser.add_argument("--computed_distances", type=str, help="Pickle file to store computed pairwise distances")
parser.add_argument("--output", type=str, help="TSV file where intra and inter dataset distances will be written")
parser.add_argument("--cosine", action="store_true", help="if set, calculate cosine diffs")
parser.add_argument("--magnitude", action="store_true", help="if set, calculate magnitudes")
parser.add_argument("--computed_norms", type=str, help="Pickle file to store computed norms")
parser.add_argument("--mag_output", type=str, help="Pickle file to store computed norms")
args = parser.parse_args()


datasets_of_interest = [x.strip() for x in open(args.datasets)]
if args.computed_distances is None or not os.path.exists(args.computed_distances):
    print("Reading P3 data")
    p3_data_filename = args.p3_data
    p3_dataset_indices_filename = args.p3_indices
    text_data = {x: [] for x in datasets_of_interest}
    p3_data_ptr = open(p3_data_filename, "rt")
    p3_indices_ptr = open(p3_dataset_indices_filename, "rt")
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
    for dataset_name in datasets_of_interest:
        print(f"\t{dataset_name}\t{len(text_data[dataset_name])}")
    p3_data_ptr.close()
    p3_indices_ptr.close()


    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.cuda()
    parameters_of_interest = []
    for name, parameter in model.named_parameters():
        if name.startswith("encoder.final_layer_norm"):
            #if 'layer_norm' in name.lower() or 'layernorm' in name.lower():
            parameters_of_interest.append((name, parameter))


    print(f"Computing gradients on {args.model}")
    #print(f"Computing gradients only on {args.encoder_block_name} and the final layer norm weight")

    all_dataset_gradients = []
    norms = []
    for dataset_name in tqdm(datasets_of_interest):
        instances = text_data[dataset_name]
        for instance in tqdm(instances):
            inputs = tokenizer.encode(instance["input"], return_tensors="pt").cuda()
            targets = tokenizer.encode(instance["target"], return_tensors="pt").cuda()
            model_outputs = model(input_ids=inputs, labels=targets, return_dict=True)
            loss = model_outputs['loss']
            loss.backward(inputs=[p for n, p in parameters_of_interest])

            gradients = torch.cat([p.grad.flatten() for _, p in parameters_of_interest]).detach().cpu().numpy()
            #all_dataset_gradients.append(gradients)
            norms.append(np.linalg.norm(gradients, axis=-1))
            model.zero_grad()
        
    #all_gradients = numpy.stack(all_dataset_gradients)
    norms = numpy.stack(norms)
    #print(f"Done computing gradients. The size of the matrix is {all_gradients.shape}")

    if args.run_pca: 
        print("Running PCA")
        pca = PCA(n_components=args.num_pca_components, random_state=0)
        all_gradients = pca.fit_transform(all_gradients)
        print(f"Done running PCA. The size of the gradients matrix is {all_gradients.shape}")

    if args.cosine:
        print("Computing cosine distances")
        distances = 1 - fastdist.cosine_pairwise_distance(all_gradients, return_matrix=True)
        if args.computed_distances is not None:
            with open(args.computed_distances, "wb") as outfile:
                pickle.dump(distances, outfile)
    if args.magnitude:
        print("Computing magnitudes")
        #norms = np.linalg.norm(all_gradients, axis=-1) # take norms
        if args.computed_norms is not None:
            with open(args.computed_norms, "wb") as outfile:
                pickle.dump(norms, outfile)
else:
    with open(args.computed_distances, "rb") as infile:
        distances = pickle.load(infile)
    if args.computed_norms:
        with open(args.computed_norms, "rb") as infile:
            distances = pickle.load(infile)
    


dataset_index_ranges = {
        name: (
            i * args.max_instances_per_dataset,
            (i + 1) * args.max_instances_per_dataset
            )
        for i, name in enumerate(datasets_of_interest)
        }

if args.cosine:
    with open(args.output, "w") as outfile:
        print("Intra dataset averages", file=outfile)
        for dataset_name in datasets_of_interest:
            i, j = dataset_index_ranges[dataset_name]
            print(f"{dataset_name}\t{distances[i:j, i:j].mean()}", file=outfile)

        print("\nInter dataset averages", file=outfile)
        print("\t".join([""] + datasets_of_interest[1:]), file=outfile)
        for d1 in range(len(datasets_of_interest) - 1):
            d1_distances = []
            dataset_name1 = datasets_of_interest[d1]
            for d2 in range(d1 + 1, len(datasets_of_interest)):
                dataset_name2 = datasets_of_interest[d2]
                i1, j1 = dataset_index_ranges[dataset_name1]
                i2, j2 = dataset_index_ranges[dataset_name2]
                d1_distances.append(distances[i1:j1, i2:j2].mean())
            padding = [""] * (len(datasets_of_interest) - len(d1_distances) - 1)
            output_line = "\t".join([dataset_name1] + padding + [str(d) for d in d1_distances])
            print(output_line, file=outfile)

if args.magnitude:
    with open(args.mag_output, "w") as outfile:
        print("intra-dataset magnitude variance", file=outfile)
        for dataset_name in datasets_of_interest:
            i, j = dataset_index_ranges[dataset_name]
            print(f"{dataset_name}\t{norms[i:j].var()}", file=outfile)

        print("\nInter-dataset variance", file=outfile)
        print("\t".join([""] + datasets_of_interest[1:]), file=outfile)
        for d1 in range(len(datasets_of_interest) - 1):
            d1_norms = []
            dataset_name1 = datasets_of_interest[d1]
            for d2 in range(d1 + 1, len(datasets_of_interest)):
                dataset_name2 = datasets_of_interest[d2]
                i1, j1 = dataset_index_ranges[dataset_name1]
                i2, j2 = dataset_index_ranges[dataset_name2]
                # '.r_' means we pick up two ranges in the same dim
                d1_norms.append(np.concatenate([norms[i1:j1], norms[i2:j2]], axis=0).var())
            padding = [""] * (len(datasets_of_interest) - len(d1_norms) - 1)
            output_line = "\t".join([dataset_name1] + padding + [str(d) for d in d1_norms])
            print(output_line, file=outfile)
