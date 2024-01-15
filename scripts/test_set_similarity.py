"""
Analysis script for comparing the similarity of the test set to the training set.
Currently looking at correlation b/w mean distance and correctness (binned).
"""
import json
import os
import pickle
import random
from functools import partial
from multiprocessing import Pool
from collections import defaultdict, Counter
import argparse
import faiss

import numpy as np
import scipy
import torch
from fastdist import fastdist
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from matplotlib import pyplot as plt

random_gen = random.Random(42)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--use_eos", action="store_true"
)
parser.add_argument(
    "--use_sgpt", action="store_true"
)
parser.add_argument(
    "--model_name", type=str  # "/net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B
)
# mark if the model is a sentence transformer
parser.add_argument(
    "--sentence_transformer", action="store_true"
)
parser.add_argument(
    "--encoded_data_path", type=str # "data/vectors/camel_encodings_sgpt.npy"
)
parser.add_argument(
    "--encoded_data_metadata_path", type=str # "data/vectors/camel_encodings_sgpt_metadata.pkl"
)
parser.add_argument(
    "--subsample", type=int, default=0
)
parser.add_argument(
    "--include_response", action="store_true"
)
parser.add_argument(
    "--subsample_test_points", type=int, default=0
)
parser.add_argument(
    "--distance_metric", choices=["cosine", "ip", "l2", "cosine_thresh"], default="cosine"
)
parser.add_argument(
    "--cosine_thresh", type=float, default=0.8, help="threshold for cosine similarity if using cosine_thresh distance metric. Keep only points greater than this value."
)
parser.add_argument(
    "--use_fewshot", action="store_true", help="use fewshot examples in prompt."
)
parser.add_argument(
    "--use_loss", action="store_true", help="Measure correl with loss instead of correctness."
)
parser.add_argument(
    "--base_folder", type=str, help="File to get base performance from.", default=None
)
parser.add_argument(
    "--use_flipped", action="store_true", help="Only count examples that flipped to correct after training in correctness."
)
args = parser.parse_args()


prompts = defaultdict(list)
completions = defaultdict(list)
correctness = defaultdict(list)
flipped = defaultdict(list)
base_preds = {}
base_probs = {}
bbh_range = {}
mmlu_range = {}
correct_probs = defaultdict(list)
prob_list = ["A", "B", "C", "D"]

if args.base_folder:
    for file in os.listdir(args.base_folder):
        if file.endswith(".jsonl"):
            dataset = [json.loads(data) for data in open(os.path.join(args.base_folder, file), 'r')]
            for sample in dataset:
                import re
                clean_input = re.sub(r'\n\n.*\n\n', '\n', sample['input'].replace('\ufeff', 'x'))
                base_preds[clean_input] = sample['prediction']
                if 'probs' in sample:
                    base_probs[clean_input] = sample['probs'][prob_list.index(sample['prediction'])]

for file in os.listdir(f"data/camel_results/7b/bbh_cot_{0 if not args.use_fewshot else 3}shot"):
    if file.endswith(".jsonl"):
        f = file.split(".")[0]
        dataset = [json.loads(data) for data in open(f"data/camel_results/7b/bbh_cot_{0 if not args.use_fewshot else 3}shot/{file}")]
        prompts[f] = [x["input"] for x in dataset if x['input'] in base_preds]
        completions[f] = [x["target"] for x in dataset if x['input'] in base_preds]
        correctness[f] = [x["was_correct"] for x in dataset if x['input'] in base_preds]
        flipped[f] = [True if x['prediction'] != base_preds[x['input']] and x['was_correct'] else False for x in dataset if x['input'] in base_preds]

        bbh_range[f] = (0, len(prompts[f]))
# mmlu
for file in os.listdir(f"data/camel_results/7b/mmlu_{0 if not args.use_fewshot else 5}shot"):
    if file.endswith(".jsonl"):
        f = file.split(".")[0]
        dataset = [json.loads(data) for data in open(f"data/camel_results/7b/mmlu_{0 if not args.use_fewshot else 5}shot/{file}")]
        prompts[f] += [x["input"] for x in dataset if x['input'] in base_preds]
        completions[f] += [x["target"] for x in dataset if x['input'] in base_preds]
        correctness[f] += [x["was_correct"] for x in dataset if x['input'] in base_preds]
        flipped[f] += [True if x['prediction'] != base_preds[x['input']] and x['was_correct'] else False for x in dataset if x['input'] in base_preds]
        mmlu_range[f] = (bbh_range[f][1], len(prompts[f]))
        correct_probs[f] += [x["probs"][prob_list.index(x["target"])] for x in dataset if x['input'] in base_preds]

if args.subsample_test_points > 0:
    for key in prompts:
        # select random indices
        subsampled = random_gen.sample(list(range(len(prompts[key]))), args.subsample_test_points)
        prompts[key] = [x for i, x in enumerate(prompts[key]) if i in subsampled]
        completions[key] = [x for i, x in enumerate(completions[key]) if i in subsampled]
        correctness[key] = [x for i, x in enumerate(correctness[key]) if i in subsampled]

# load model
if not args.sentence_transformer:
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name)
        .eval()
        .cuda()
        .half()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=False
    )
else:
    model = SentenceTransformer(args.model_name).eval().cuda().half()
    tokenizer = model.tokenizer

# load camel encodings
camel_encoded_data = np.load(args.encoded_data_path)
camel_metadata = pickle.load(
    open(args.encoded_data_metadata_path, "rb")
)
camels = camel_metadata["camels"]
camel_lengths = camel_metadata["camel_lengths"]

tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

# encode prompts first - assuming all prompts in same order.
prompt_encodings = {}
prompt_cache = {}
prefix = '_few_shot' if args.use_fewshot else ''
if True:#not os.path.exists(f"data/prompt_encs/{args.model_name.replace('/', '_') + prefix}-prompt_encodings.npy"):
    print("Encoding prompts...")
    with torch.inference_mode():
        # encode everything fine
        for f in prompts:
            prompt_encodings[f] = []
            for i, (prompt, completion) in tqdm(enumerate(zip(prompts[f], completions[f]))):
                prompt = f"<|user|>\n{prompt}"
                if args.include_response:
                    prompt += f"<|assistant|>\n{completion}"
                if prompt in prompt_cache:
                    encoded_prompt = prompt_cache[prompt]
                else:
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                    if args.sentence_transformer:
                        encoded_prompt = model.encode(prompt)
                    elif args.use_eos:
                        input_ids = torch.cat(
                            [input_ids, torch.ones((input_ids.size(0), 1)) * tokenizer.eos_token_id], axis=-1
                        )
                        encoded = model(input_ids.long().cuda(), output_hidden_states=True)
                        encoded_prompt = encoded.hidden_states[-1][0, -1].detach().cpu().numpy()
                    elif args.use_sgpt:
                        position_weights = (
                            torch.arange(start=1, end=input_ids.shape[1] + 1).float()
                            / torch.arange(input_ids.shape[1] + 1).sum().float()
                        )
                        encoded = model(input_ids.long().cuda(), output_hidden_states=True)
                        sentence_embedding = (
                            position_weights[:, None].cuda() * encoded.hidden_states[-1][0]
                        ).sum(0)
                        encoded_prompt = sentence_embedding.detach().cpu().numpy()
                prompt_encodings[f].append(encoded_prompt)
                prompt_cache[prompt] = encoded_prompt
        np.save(f"data/prompt_encs/{args.model_name.replace('/', '_') + prefix}-prompt_encodings.npy", np.stack(prompt_encodings))
else:
    prompt_encodings = np.load(f"data/prompt_encs/{args.model_name.replace('/', '_')}-prompt_encodings.npy")

# loss calcs
model_split, model_recovered = None, None
if args.use_loss:
    from scripts.create_llama_encodings import encode_with_messages_format
    with torch.inference_mode():
        for f in prompts:
            hf_name = 'baize'
            # if 'super-ni' in hf_name:
            #     hf_name = sni
            if model_split is None:
                model_split = AutoModelForCausalLM.from_pretrained(f'allenai/open-instruct-{hf_name}-7b')
                model_recovered = AutoModelForCausalLM.from_pretrained('/net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B')
                tokenizer = AutoTokenizer.from_pretrained(
                    f'allenai/open-instruct-{hf_name}-7b', use_fast=False
                )
                model_recovered.resize_token_embeddings(len(tokenizer))
                state_dict_recovered = model_recovered.state_dict()
                state_dict_raw = model_split.state_dict()
                for key in tqdm(state_dict_recovered):
                    state_dict_recovered[key].add_(state_dict_raw[key])
                # calculate loss over prompts
                model_recovered = model_recovered.cuda()
            for i, (prompt, completion) in tqdm(enumerate(zip(prompts[f], completions[f]))):
                if i in range(mmlu_range[f][0], mmlu_range[f][1]):
                    # looking at ARG. dont divide by original score because...
                    # that would rate .1->.15 the same as .2->.3. That feels wrong?
                    correctness[f][i] = float(correct_probs[f][i])# - float(base_probs[prompt]))
                    continue
                inp = {'messages': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': completion}]}
                inputs = encode_with_messages_format(
                    inp, tokenizer, 2048, args.include_response, False
                )
                loss = model_recovered(inputs['input_ids'][None,].cuda(), labels=inputs['labels'][None,].cuda()).loss
                correctness[f][i] = loss.detach().cpu().numpy()
            # del model_recovered
            # del state_dict_recovered, state_dict_raw
            # torch.cuda.empty_cache()


# take model off gpu now its not needed
model = model.cpu()
torch.cuda.empty_cache()

print("Finding how far our camels have to travel...🐪🐫")


all_cos_mats = []
all_cos_nns = []
all_cos_twos = []
all_corrects = []

def calc_binned_correl(cos_sixes, correctness, bins=10):
    res = scipy.stats.binned_statistic(cos_sixes, correctness, statistic='mean', bins=bins)
    print(f"Bin counts: {Counter(res.binnumber)}")
    binned_correctness = res.statistic
    binned_distance = scipy.stats.binned_statistic(cos_sixes, cos_sixes, statistic='mean', bins=bins).statistic
    # filter nans
    binned_distance = binned_distance[~np.isnan(binned_correctness)]
    binned_correctness = binned_correctness[~np.isnan(binned_correctness)]
    print("Distances:" + str(binned_distance))
    print("Correctness:" + str(binned_correctness))
    print(f"Pearson r = {scipy.stats.pearsonr(binned_correctness, binned_distance)}")
    return binned_distance, binned_correctness

counter = 0
all_neighbours = []
camel_encoded_data = camel_encoded_data.squeeze()  # to be safe
for f in camel_lengths:
    print(f"-------------- {f} --------------")
    df_data = camel_encoded_data[counter : counter + camel_lengths[f]]    
    metadata = camels[f]
    counter += camel_lengths[f]
    corrects = correctness[f]
    if args.use_flipped:
        corrects = flipped[f]
    all_corrects += corrects
    print("index creation...")
    # brute force indices are enough for the dataset sizes im looking at.
    if args.distance_metric == "l2":
        metric = faiss.METRIC_L2
    else:
        metric = faiss.METRIC_INNER_PRODUCT
    index = faiss.index_factory(df_data[0].shape[0], "Flat", metric)
    # print("sending index to gpu...")
    # index = faiss.index_cpu_to_all_gpus(index)
    print("add data to index...")
    df_data = df_data.astype(np.float32)
    if args.distance_metric == "cosine" or args.distance_metric == "cosine_thresh":
        faiss.normalize_L2(df_data)
    index.add(df_data)
    print("searching index...")
    import time
    time_start = time.time()
    query = np.stack(prompt_encodings[f], 0)
    query = query.astype(np.float32)
    if args.distance_metric == "cosine" or args.distance_metric == "cosine_thresh":
        faiss.normalize_L2(query)
    if args.distance_metric == "cosine_thresh":
        threshold = args.cosine_thresh
        lims, d, i = index.range_search(x=query, thresh=threshold)
        counts = [y - x for x, y in zip(lims, lims[1:])]
        D = np.array(counts)[:, None]
    else:
        D, I = index.search(query, 10)
        all_neighbours.append(I)
    all_cos_nns.append(D)
    print(f"search time: {time.time() - time_start}")
    print("calculating distances...")
    max_range = 11 if not args.distance_metric == "cosine_thresh" else 2
    for i in range(1, max_range):
        cos_sixes = D[:,:i].mean(1)
        print(f"--- knn-{i} ---")
        correctness[f] = np.array(correctness[f])
        # print("for bbh")
        # calc_binned_correl(cos_sixes[:bbh_range[f][1]], corrects[:bbh_range[f][1]])
        print("for mmlu")
        calc_binned_correl(cos_sixes[mmlu_range[f][0]:mmlu_range[f][1]], corrects[mmlu_range[f][0]:mmlu_range[f][1]], bins=100000)
        print("for all")
        binned_distance, binned_correctness = calc_binned_correl(cos_sixes, corrects, bins=100000)
        # plot for viz
        plt.scatter(binned_distance, binned_correctness, label=f"knn-{i}")
    plt.legend()
    plt.savefig(f"plots/{args.model_name.replace('/', '_')}-sgpt-knn-{f}-{args.distance_metric}.png")
    plt.close('all')
    del index
    torch.cuda.empty_cache()

all_neighbours = np.concatenate(all_neighbours, axis=0)
np.save(open('neighbours_llama_7b.npy', 'wb'), all_neighbours)

print("-------------- all results --------------")
for i in range(1, 11):
    print(f"--- knn-{i} ---")
    all_mean_dists = np.concatenate(all_cos_nns, axis=0)[:, :i].mean(1)
    binned_correctness = scipy.stats.binned_statistic(
        all_mean_dists, all_corrects, statistic="mean", bins=100000
    ).statistic
    binned_distance = scipy.stats.binned_statistic(
        all_mean_dists, all_mean_dists, statistic="mean", bins=100000
    ).statistic
    print(f"Bin counts: {Counter(scipy.stats.binned_statistic(all_mean_dists, all_corrects, statistic='mean', bins=100000).binnumber)}")

    # filter nans
    binned_distance = binned_distance[~np.isnan(binned_correctness)]
    binned_correctness = binned_correctness[~np.isnan(binned_correctness)]
    print("Distances:" + str(binned_distance))
    print("Correctness:" + str(binned_correctness))
    print(f"Pearson r = {scipy.stats.pearsonr(binned_distance, binned_correctness)}")
    plt.scatter(binned_distance, binned_correctness, label=f"knn-{i}")
plt.legend()
plt.savefig(f"plots/{args.model_name.replace('/', '_')}-sgpt-knns--{args.distance_metric}-all.png")
plt.close('all')
