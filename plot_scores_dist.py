import json
import argparse
import statistics
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("filenames", type=str)
args = parser.parse_args()

all_scores = []
for filename in args.filenames.split(","):
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))

    scores = [x['influence_score'] for x in data]
    print(filename)
    # get mean score
    print(statistics.mean(scores))
    # get standard deviation
    print(statistics.stdev(scores))
    print(max(scores))
    all_scores.append(scores)
plt.rcParams.update({'font.size': 16})  # Adjust font size globally
evals = ['AlpacaEval', 'GSM8k', 'Codex', 'Squad', 'MMLU', 'TydiQA', 'BBH']
# aggregate non-alpacaeval scores and non-codex scores
plt.hist(all_scores[0], bins=50, alpha=0.5, label=evals[0])
plt.hist(all_scores[1], bins=50, alpha=0.5, label=evals[1])

# plt.hist(non_alpaca_codex_scores, bins=50, alpha=0.5, label='Other', density=True)
plt.ylabel('Frequency')
plt.xlabel('Cosine Sim.')
plt.legend()
plt.tight_layout()
plt.savefig('scores_dist.pdf')