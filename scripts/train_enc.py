from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import itertools
import random
import os
import json
from scripts.create_llama_encodings import encode_with_messages_format
from tqdm import tqdm
from scipy.spatial.distance import cosine

random_gen = random.Random(42)
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

train_model = False

inputs, correctness, flipped, mmlu_range, correct_probs = {}, {}, {}, {}, {}
base_preds, base_probs = {}, {}

print("Setting up data!")
# base preds
prob_list = ["A", "B", "C", "D"]
for file in os.listdir('data/base_results'):
    if file.endswith(".jsonl"):
        dataset = [json.loads(data) for data in open(os.path.join('data/base_results', file), 'r')]
        for sample in dataset:
            import re
            clean_input = re.sub(r'\n\n.*\n\n', '\n', sample['input'].replace('\ufeff', 'x'))
            base_preds[clean_input] = sample['prediction']
            if 'probs' in sample:
                base_probs[clean_input] = sample['probs'][prob_list.index(sample['prediction'])]

# test points
for file in os.listdir(f"data/camel_results/7b/mmlu_5shot"):
    if file.endswith(".jsonl"):
        f = file.split(".")[0]
        dataset = [json.loads(data) for data in open(f"data/camel_results/7b/mmlu_5shot/{file}")]
        inputs[f] = [f'<|user|>\n{x["input"]}\n<|assistant|>{x["target"]}' for x in dataset if x['input'] in base_preds]
        correctness[f] = [x["was_correct"] for x in dataset if x['input'] in base_preds]
        flipped[f] = [True if x['prediction'] != base_preds[x['input']] and x['was_correct'] else False for x in dataset if x['input'] in base_preds]
        correct_probs[f] = [x["probs"][prob_list.index(x["target"])] for x in dataset if x['input'] in base_preds]

# train points
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
for filename in camel_datasets:
    with open(f"{path}/{filename}/{filename}_data.jsonl", "r") as f:
        camels[filename] = []
        for line in f:
            sample = json.loads(line.strip())
            camels[filename].append(f'<|user|>\n{sample["messages"][0]["content"]}\n<|assistant|>{sample["messages"][1]["content"]}')

print("Constructing data pairs...")

# construct dataset
samples = set()
for f in camels:
    train_samples = camels[f]
    test_samples = inputs[f]
    correct = correct_probs[f]
    # cant do all pairs, too big!
    starting_size = len(samples)
    while len(samples) < 10000 + starting_size:
        train = random_gen.choice(train_samples)
        test_idx = random_gen.choice(range(len(test_samples)))
        samples.add((train, test_samples[test_idx], correct[test_idx]))

samples = list(samples)
print("Generating train/test splits!")

# train/test split
random_gen.shuffle(samples)
train_samples = samples[:int(len(samples) * 0.8)]
test_samples = samples[int(len(samples) * 0.8):]

train_samples = [InputExample(texts=[x[0], x[1]], label=float(x[2])) for x in train_samples]
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)

evaluator = evaluation.EmbeddingSimilarityEvaluator(
    [x[0] for x in test_samples],
    [x[1] for x in test_samples],
    [float(x[2]) for x in test_samples]
)

train_loss = losses.CosineSimilarityLoss(model)



if train_model:
    print("training!")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=100,
        evaluator=evaluator,
        evaluation_steps=500,
        output_path='train_enc_output'
    )
else:
    model = SentenceTransformer('train_enc_output')
    dist_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    print("evaluating!")
    for x, y, score in test_samples:
        # get model prediction
        x_enc = model.encode(x)
        y_enc = model.encode(y)
        x_enc_dist = dist_model.encode(x)
        y_enc_dist = dist_model.encode(y)
        pred_score = 1 - cosine(x_enc, y_enc)
        dist_score = 1 - cosine(x_enc_dist, y_enc_dist)
        x = x.replace('\n', '\\n').replace('\t', ' ')
        y = y.replace('\n', '\\n').replace('\t', ' ')
        print(f"{x}\t{y}\t{pred_score}\t{score}\t{dist_score}")

print("Done training!")