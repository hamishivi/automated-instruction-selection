import pdb
import pickle
from scipy.stats import pearsonr, spearmanr

alpaca = pickle.load(open("pickles/correct_alpaca_influences.pkl", "rb"))
mmlu = pickle.load(open("pickles/correct_mmlu_influences.pkl", "rb"))
squad = pickle.load(open("pickles/correct_squad_influences.pkl", "rb"))
norm_alpaca = pickle.load(open("pickles/normalized_alpaca_influences.pkl", "rb"))

# okay, for each train instance we will compute the mean score across the dicts.
train_instances = list(alpaca[0].keys())


def compute_average_train_scores(d):
    scores = {}
    for _, train_dict in d.items():
        for train_instance, score in train_dict.items():
            train_instance = train_instance.item()
            score = score.item()
            if train_instance not in scores:
                scores[train_instance] = []
            scores[train_instance].append(score)
    return {k: sum(v) / len(v) for k, v in scores.items()}


average_alpaca = compute_average_train_scores(alpaca)
average_mmlu = compute_average_train_scores(mmlu)
average_squad = compute_average_train_scores(squad)
average_norm_alpaca = compute_average_train_scores(norm_alpaca)


def compute_max_train_scores(d):
    scores = {}
    for _, train_dict in d.items():
        for train_instance, score in train_dict.items():
            train_instance = train_instance.item()
            score = score.item()
            if train_instance not in scores:
                scores[train_instance] = []
            scores[train_instance].append(score)
    return {k: min(v) for k, v in scores.items()}


max_alpaca = compute_max_train_scores(alpaca)
max_mmlu = compute_max_train_scores(mmlu)
max_squad = compute_max_train_scores(squad)
max_norm_alpaca = compute_max_train_scores(norm_alpaca)

alpaca_scores = [max_alpaca[i] for i in range(len(max_alpaca))]
mmlu_scores = [max_mmlu[i] for i in range(len(max_mmlu))]
squad_scores = [max_squad[i] for i in range(len(max_squad))]
norm_alpaca_scores = [max_norm_alpaca[i] for i in range(len(max_norm_alpaca))]

print("pearson:")
print("alpaca mmlu", end=" ")
print(pearsonr(alpaca_scores, mmlu_scores))
print("alpaca squad", end=" ")
print(pearsonr(alpaca_scores, squad_scores))
print("alpaca norm_alpaca", end=" ")
print(pearsonr(alpaca_scores, norm_alpaca_scores))
print("mmlu squad", end=" ")
print(pearsonr(mmlu_scores, squad_scores))
print("mmlu norm_alpaca", end=" ")
print(pearsonr(mmlu_scores, norm_alpaca_scores))
print("squad norm_alpaca", end=" ")
print(pearsonr(squad_scores, norm_alpaca_scores))

print("spearman:")
print("alpaca mmlu", end=" ")
print(spearmanr(alpaca_scores, mmlu_scores))
print("alpaca squad", end=" ")
print(spearmanr(alpaca_scores, squad_scores))
print("alpaca norm_alpaca", end=" ")
print(spearmanr(alpaca_scores, norm_alpaca_scores))
print("mmlu squad", end=" ")
print(spearmanr(mmlu_scores, squad_scores))
print("mmlu norm_alpaca", end=" ")
print(spearmanr(mmlu_scores, norm_alpaca_scores))
print("squad norm_alpaca", end=" ")
print(spearmanr(squad_scores, norm_alpaca_scores))

pdb.set_trace()
