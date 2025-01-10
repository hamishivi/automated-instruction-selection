import argparse
import json
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument("--eval_file", type=str)
parser.add_argument("--cosine_scores", type=str)
args = parser.parse_args()

eval_samples = []
with open(args.eval_file, "r", encoding='utf-8') as f:
    for line in f:
        sample = json.loads(line)
        eval_samples.append(sample)


cosine_scores = []
with open(args.cosine_scores, "r", encoding='utf-8') as f:
    for idx, line in enumerate(f):
        score = json.loads(line)
        eval_samples[idx]["cosine_score"] = score

# run through examples, get top score, get metric
cosine_scores = []
metrics = []
for sample in eval_samples:
    cosine_score = sorted(sample["cosine_score"], reverse=True)[0]
    metric = sample["score"]
    cosine_scores.append(cosine_score)
    metrics.append(metric)

# compute correlation
correlation = spearmanr(cosine_scores, metrics)
print(correlation)
