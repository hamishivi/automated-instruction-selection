import argparse
import pickle
from matplotlib import pyplot as plt
import os

from transformers import AutoTokenizer
from datasets import load_dataset

from minimal_multitask.utils import encode_with_messages_format
from minimal_multitask.data import DATASETS


def visualization(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.train_dataset == "alpaca":
        train_dataset = load_dataset(
            "json", data_files="data/camel_datasets/stanford_alpaca/stanford_alpaca_data.jsonl"
        )
        train_dataset = train_dataset.map(lambda x: encode_with_messages_format(x, tokenizer, 512, True, False))
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    train_dataset = train_dataset["train"]

    if args.eval_dataset in DATASETS:
        test_dataset = DATASETS[args.eval_dataset](tokenizer).get_all_test_prompts()
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    train_len_mapping = {idx: len([tok for tok in x["labels"] if tok != -100]) for idx, x in enumerate(train_dataset)}
    test_len_mapping = {idx: len([tok for tok in x["labels"] if tok != -100]) for idx, x in enumerate(test_dataset)}

    influence_scores = pickle.load(open(args.score_file, "rb"))
    if args.test_length_cutoff >= 0:
        test_filtered = [idx for idx in influence_scores.keys() if test_len_mapping[idx] > args.length_cutoff]
    else:
        test_filtered = influence_scores.keys()
    # test_filtered = [list(test_filtered)[471]]
    aggregate_scores = {}
    for test_idx in test_filtered:
        inf = influence_scores[test_idx]
        # assert len(inf) == len(train_len_mapping)
        for k, v in inf.items():
            if k not in aggregate_scores:
                aggregate_scores[k] = []
            aggregate_scores[k].append(v)

    def sample_is_nop(idx):
        if train_len_mapping[idx] <= 1:
            return True
        if "nooutput" in tokenizer.decode(train_dataset[idx]["input_ids"]).lower():
            return True
        return False

    is_nop = [sample_is_nop(idx) for idx in range(len(train_dataset))]
    aggregate_scores = {k: sum(v) / len(v) for k, v in aggregate_scores.items()}
    score_rank = sorted(aggregate_scores.items(), key=lambda x: x[1], reverse=True)
    plt.scatter(
        list(range(len(score_rank))),
        [train_len_mapping[i[0]] for i in score_rank],
        c=[0 if is_nop[i[0]] else 1 for i in score_rank],
        alpha=0.5,
    )
    plt.xlabel("Influence Rank")
    plt.ylabel("Length")
    plt.savefig(
        os.path.join(
            args.save_dir, f'rank_testlen{args.test_length_cutoff}_{args.score_file.split("/")[-1].split(".")[0]}.png'
        )
    )
    plt.clf()
    # plot: influence against length
    plt.scatter(
        [aggregate_scores[i[0]] for i in score_rank],
        [train_len_mapping[i[0]] for i in score_rank],
        c=[0 if is_nop[i[0]] else 1 for i in score_rank],
        alpha=0.5,
    )
    plt.xlabel("Influence Score")
    plt.ylabel("Length")
    plt.savefig(
        os.path.join(
            args.save_dir,
            f'influence_testlen{args.test_length_cutoff}_{args.score_file.split("/")[-1].split(".")[0]}.png',
        )
    )
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", type=str, choices=["alpaca"], default="alpaca")
    parser.add_argument("--eval_dataset", type=str, choices=DATASETS.keys(), default="alpacafarm")
    parser.add_argument("--score_file", type=str, default="results/llama_7b/correct_squad_influences.pkl")
    parser.add_argument("--train_length_cutoff", type=int, help="Only use train example above this length", default=-1)
    parser.add_argument("--test_length_cutoff", type=int, help="Only use test example above this length", default=-1)
    parser.add_argument("--filter_noise", action="store_true")
    parser.add_argument("--tokenizer", type=str, default="huggyllama/llama-7b")
    parser.add_argument(
        "--save_dir", type=str, help="Directory to save visualizations", default="results/llama_7b/visualizations"
    )

    args = parser.parse_args()

    visualization(args)
