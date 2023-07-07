'''
Analysis script for comparing the similarity of the test set to the training set.
Currently looking at correlation b/w mean distance and correctness (binned).
'''
import json
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import pickle
import random
from datasets import load_dataset
from fastdist import fastdist
from statistics import mean, median, stdev
from datasets import load_dataset
import pandas as pd
import scipy
import os
from scipy.stats import wasserstein_distance
from multiprocessing import Pool



random_gen = random.Random(42)

use_eos = True
use_sgpt = False
eval_dataset = "bbh"

subsample = 1000


# lets start with just alpacafarm
if eval_dataset == "alpacafarm":
    prompts = [x["instruction"] for x in load_dataset("hamishivi/alpaca-farm-davinci-003-2048-token", split="train")]
    # we could use the gpt generations?
    completions = [x["output"] for x in load_dataset("hamishivi/alpaca-farm-davinci-003-2048-token", split="train")]
elif eval_dataset == "bbh":
    prompts = {}
    completions = {}
    correctness = {}
    for file in os.listdir("bbh_cot_datasets"):
        if file.endswith(".jsonl"):
            f = file.split(".")[0]
            dataset = [json.loads(data) for data in open(f"bbh_cot_datasets/{file}")]
            prompts[f] = [x["input"] for x in dataset]
            completions[f] = [x["target"] for x in dataset]
            correctness[f] = [x["was_correct"] for x in dataset]
    # mmlu
    for file in os.listdir("mmlu_5shot_datasets"):
       if file.endswith(".jsonl"):
           f = file.split(".")[0]
           dataset = [json.loads(data) for data in open(f"mmlu_5shot_datasets/{file}")]
           prompts[f] += [x["input"] for x in dataset]
           completions[f] += [x["target"] for x in dataset]
           correctness[f] += [x["was_correct"] for x in dataset]

# subsample test instances to speed things up
if subsample > 0:
    for key in prompts:
        # select random indices
        subsampled = random_gen.sample(list(range(len(prompts[key]))), subsample)
        prompts[key] = [x for i, x in enumerate(prompts[key]) if i in subsampled]
        completions[key] = [x for i, x in enumerate(completions[key]) if i in subsampled]
        correctness[key] = [x for i, x in enumerate(correctness[key]) if i in subsampled]

# load model
model = AutoModelForCausalLM.from_pretrained('/net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B').eval().cuda().half()
tokenizer = AutoTokenizer.from_pretrained('/net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B', use_fast=False)

# load camel encodings
camel_encoded_data = np.load("data/vectors/camel_encodings_prompt_only_eos_all.npy")
camel_metadata = pickle.load(open("data/vectors/camel_encodings_prompt_only_eos_all_metadata.pkl", "rb"))
camels = camel_metadata["camels"]
camel_lengths = camel_metadata["camel_lengths"]

# encode prompts first - assuming all prompts in same order.
prompt_encodings = []
with torch.inference_mode():
    for prompt, completion in tqdm(zip(prompts['baize'], completions['baize'])):
        prompt = f"<|user|>\n{prompt}\n"#<|assistant|>\n{completion}"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        if use_eos:    
            input_ids = torch.cat([input_ids, torch.ones((input_ids.size(0), 1))*tokenizer.eos_token_id], axis=-1)
            encoded = model(input_ids.long().cuda(), output_hidden_states=True)
            encoded_prompt = encoded.hidden_states[-1][0,-1].detach().cpu().numpy()
        elif use_sgpt:
            position_weights = torch.arange(start=1, end=input_ids.shape[1]+1).float() / torch.arange(input_ids.shape[1]+1).sum().float()
            encoded = model(input_ids.long().cuda(), output_hidden_states=True)
            sentence_embedding = (position_weights[:,None].cuda() * encoded.hidden_states[-1][0]).sum(0)
            encoded_prompt = sentence_embedding.detach().cpu().numpy()
        prompt_encodings.append(encoded_prompt)

def euc_dist(enc_data, vec):
    return fastdist.vector_to_matrix_distance(vec.astype(np.float64), enc_data.astype(np.float64), fastdist.euclidean, "euclidean")
def cos_dist(enc_data, vec):
    return fastdist.vector_to_matrix_distance(vec.astype(np.float64), enc_data.astype(np.float64), fastdist.cosine, "cosine")

from functools import partial
def get_min_dist(encoded_prompts, encoded_data):
    # since this is all on cpu, we can use multiprocessing to speed it up.
    pool = Pool(64)
    euc_distances = list(tqdm(pool.imap(partial(euc_dist, encoded_data), encoded_prompts), total=len(encoded_prompts)))
    cos_distances = list(tqdm(pool.imap(partial(cos_dist, encoded_data), encoded_prompts), total=len(encoded_prompts)))
    return np.stack(euc_distances, 0), np.stack(cos_distances, 0)
    
print("Finding how far our camels have to travel...🐪🐫")

from matplotlib import pyplot as plt

# tsne the data and eval data
from sklearn.manifold import TSNE

all_euc_mats = []
all_corrects = []

counter = 0
for f in camel_lengths:
    df_data = camel_encoded_data[counter:counter+camel_lengths[f]]
    metadata = camels[f]
    counter += camel_lengths[f]
    euc_mat, cos_mat = get_min_dist(np.stack(prompt_encodings, 0), df_data)
    euc_mins = euc_mat.min(axis=1).tolist()
    cos_mins = cos_mat.max(axis=1).tolist()
    
    # Create a DataFrame from the NumPy matrix
    # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 3), sharex=True)
    # for i, row in enumerate(euc_mat):
    #     axes[0 if correctness[f][i] else 1].hist(row, 1000, alpha=0.3, color='blue' if correctness[f][i] else 'red')
    # plt.savefig(f"tmp/euc_hist_{f}.png")
    # plt.close()
    # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 3), sharex=True)
    # for i, row in enumerate(cos_mat):
    #     axes[0 if correctness[f][i] else 1].hist(row, 1000, alpha=0.3, color='blue' if correctness[f][i] else 'red')
    # plt.savefig(f"tmp/cos_hist_{f}.png")
    # plt.close()
    # binned stats calc
    binned_correctness = scipy.stats.binned_statistic(euc_mat.mean(1), correctness[f], statistic='mean', bins=100).statistic
    binned_distance = scipy.stats.binned_statistic(euc_mat.mean(1), euc_mat.mean(1), statistic='mean', bins=100).statistic
    # filter nans
    binned_distance = binned_distance[~np.isnan(binned_correctness)]
    binned_correctness = binned_correctness[~np.isnan(binned_correctness)]
    all_euc_mats += euc_mat.mean(1).tolist()
    all_corrects += correctness[f]
    # correlation between correctness and distances
    print(f"{f} {scipy.stats.pearsonr(binned_correctness, binned_distance)}")

print('-------------- all results --------------')
binned_correctness = scipy.stats.binned_statistic(all_euc_mats, all_corrects, statistic='mean', bins=100).statistic
binned_distance = scipy.stats.binned_statistic(all_euc_mats, all_euc_mats, statistic='mean', bins=100).statistic
# filter nans
binned_distance = binned_distance[~np.isnan(binned_correctness)]
binned_correctness = binned_correctness[~np.isnan(binned_correctness)]
print(f"{scipy.stats.pearsonr(binned_distance, binned_correctness)}")
import pdb; pdb.set_trace()