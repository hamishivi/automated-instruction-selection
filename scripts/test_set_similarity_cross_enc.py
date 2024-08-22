"""
version of my test similarity script specifically for cross-encoders.
"""
import json
import os
import random
from collections import defaultdict, Counter
import argparse

import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from sentence_transformers import CrossEncoder

random_gen = random.Random(42)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)  # cross-encoder/stsb-roberta-large is best
parser.add_argument("--subsample", type=int, default=0)
parser.add_argument("--include_response", action="store_true")
parser.add_argument("--subsample_test_points", type=int, default=0)
parser.add_argument("--bsz", type=int, default=128)
args = parser.parse_args()


prompts = defaultdict(list)
completions = defaultdict(list)
correctness = defaultdict(list)
for file in os.listdir("data/camel_results/bbh_cot_0shot"):
    if file.endswith(".jsonl"):
        f = file.split(".")[0]
        dataset = [json.loads(data) for data in open(f"data/camel_results/bbh_cot_0shot/{file}")]
        prompts[f] = [x["input"] for x in dataset]
        completions[f] = [x["target"] for x in dataset]
        correctness[f] = [x["was_correct"] for x in dataset]
# mmlu
for file in os.listdir("data/camel_results/mmlu_0shot"):
    if file.endswith(".jsonl"):
        f = file.split(".")[0]
        dataset = [json.loads(data) for data in open(f"data/camel_results/mmlu_0shot/{file}")]
        prompts[f] += [x["input"] for x in dataset]
        completions[f] += [x["target"] for x in dataset]
        correctness[f] += [x["was_correct"] for x in dataset]

if args.subsample_test_points > 0:
    for key in prompts:
        # select random indices
        subsampled = random_gen.sample(list(range(len(prompts[key]))), args.subsample_test_points)
        prompts[key] = [x for i, x in enumerate(prompts[key]) if i in subsampled]
        completions[key] = [x for i, x in enumerate(completions[key]) if i in subsampled]
        correctness[key] = [x for i, x in enumerate(correctness[key]) if i in subsampled]

model = CrossEncoder(args.model_name, device="cuda")
tokenizer = model.tokenizer

# load camel data
camel_datasets = [
    "baize",
    "code_alpaca",
    "cot",
    "dolly",
    "flan_v2",
    "gpt4_alpaca",
    "oasst1",
    "self_instruct",
    "sharegpt",
    "stanford_alpaca",
    "super_ni",
    "unnatural_instructions",
]

path = "/net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/camel_datasets"

camels = {}
camel_lengths = {}
for filename in camel_datasets:
    with open(f"{path}/{filename}/{filename}_data.jsonl", "r") as f:
        camels[filename] = [json.loads(x.strip())["messages"][:2] for x in f]
        if not args.include_response:
            camels[filename] = [f"<|user|>\n{x[0]}" for x in camels[filename]]
        else:
            camels[filename] = [f"<|user|>\n{x[0]}\n<|assistant|>\n{x[1]}" for x in camels[filename]]

        camel_lengths[filename] = len(camels[filename])

# load prompt data
test_prompts = []
with torch.inference_mode():
    for prompt, completion in tqdm(zip(prompts["baize"], completions["baize"])):
        prompt = f"<|user|>\n{prompt}"
        if args.include_response:
            prompt += f"<|assistant|>\n{completion}"
        test_prompts.append(prompt)


print("Finding how far our camels have to travel...🐪🐫")


all_cos_nns = []
all_corrects = []
counter = 0
for f in camel_lengths:
    print(f"-------------- {f} --------------")
    df_data = camels[f][counter: counter + camel_lengths[f]]
    counter += camel_lengths[f]
    all_corrects += correctness[f]
    # loop through all pairs and get predictions
    all_scores = []
    for prompt in test_prompts:
        pairs = []
        for sample in df_data:
            pairs.append((sample, prompt))
        scores = model.predict(pairs, show_progress_bar=True, batch_size=args.bsz)
        top_10 = np.sort(scores)[:10]
        all_scores.append(top_10)
        all_cos_nns.append(top_10)
    all_scores = np.array(all_scores)
    print("calculating distances...")
    for i in range(1, 11):
        cos_sixes = all_scores[:, :i].mean(1)
        print(f"--- knn-{i} ---")
        res = scipy.stats.binned_statistic(cos_sixes, correctness[f], statistic="mean", bins=10)
        print(f"Bin counts: {Counter(res.binnumber)}")
        binned_correctness = res.statistic
        binned_distance = scipy.stats.binned_statistic(cos_sixes, cos_sixes, statistic="mean", bins=10).statistic
        # filter nans
        binned_distance = binned_distance[~np.isnan(binned_correctness)]
        binned_correctness = binned_correctness[~np.isnan(binned_correctness)]
        print("Distances:" + str(binned_distance))
        print("Correctness:" + str(binned_correctness))
        # correlation between correctness and distances
        print(f"Pearson r = {scipy.stats.pearsonr(binned_correctness, binned_distance)}")
        # plot for viz
        plt.scatter(binned_distance, binned_correctness, label=f"knn-{i}")
    plt.legend()
    plt.savefig(f"plots/{args.model_name.replace('/', '_')}-cross-encoder-knn-{f}.png")
    plt.close("all")

print("-------------- all results --------------")
for i in range(1, 11):
    print(f"--- knn-{i} ---")
    all_mean_dists = np.concatenate(all_cos_nns, axis=0)[:, :i].mean(1)
    binned_correctness = scipy.stats.binned_statistic(
        all_mean_dists, all_corrects, statistic="mean", bins=25
    ).statistic
    binned_distance = scipy.stats.binned_statistic(all_mean_dists, all_mean_dists, statistic="mean", bins=25).statistic
    print(
        f"Bin counts: {Counter(scipy.stats.binned_statistic(all_mean_dists, all_corrects, statistic='mean', bins=25).binnumber)}"
    )

    # filter nans
    binned_distance = binned_distance[~np.isnan(binned_correctness)]
    binned_correctness = binned_correctness[~np.isnan(binned_correctness)]
    print("Distances:" + str(binned_distance))
    print("Correctness:" + str(binned_correctness))
    print(f"Pearson r = {scipy.stats.pearsonr(binned_distance, binned_correctness)}")
    plt.scatter(binned_distance, binned_correctness, label=f"knn-{i}")
plt.legend()
plt.savefig(f"plots/{args.model_name.replace('/', '_')}-cross-encoder-knns-all.png")
plt.close("all")
