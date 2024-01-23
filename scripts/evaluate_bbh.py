"""
Evaluate a hf model on BBH (a subset of big bench)
"""
import argparse

import evaluate
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

parser = argparse.ArgumentParser(description="Evaluate a hf model on BBH (a subset of big bench)")
parser.add_argument("--model_name", type=str, default="google/t5-xl-lm-adapt", help="model name")
parser.add_argument(
    "--tokenizer_name",
    type=str,
    default=None,
    help="tokenizer name (defaults to model_name if not specified)",
)
parser.add_argument("--batch_size", type=int, default=8, help="batch size")
parser.add_argument(
    "--allennlp_weight",
    type=str,
    default=None,
    help="Location of allennlp weight. If specified, the model name is just "
    "used to load config, and the weights come from this file.",
)
args = parser.parse_args()

data_subsets = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]

model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
if args.allennlp_weight is not None:
    model.load_state_dict(
        {k.replace("transformer.", ""): v for k, v in torch.load(args.allennlp_weight).items()}
    )
tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name
)
metric = evaluate.load("exact_match")
# tokenize inputs

model = model.cuda()


def evaluate_samples(examples, prompt):
    with torch.inference_mode():
        dataloader = torch.utils.data.DataLoader(
            examples, batch_size=args.batch_size, shuffle=False
        )
        for batch in tqdm(dataloader):
            inputs = [f"\n{inp}\nAnswer: " for inp in batch["input"]]
            # dynamic adding of few-shot examples
            for i, inp in enumerate(inputs):
                few_shot_idx = 1
                while (
                    len(
                        tokenizer(prompt[0] + inp)["input_ids"]
                        + tokenizer(prompt[few_shot_idx])["input_ids"]
                    )
                    < 2048
                ):
                    inp = f"\n{prompt[few_shot_idx]}" + inp
                    few_shot_idx += 1
                    # stop at 3 shots
                    if few_shot_idx == len(prompt):
                        break
                inputs[i] = (prompt[0] + "\n" + inp).strip()
            # print(inputs)
            inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=False)

            # most bbh outputs are very short
            outputs = (
                model.generate(
                    inputs["input_ids"].cuda(),
                    attention_mask=inputs["attention_mask"].cuda(),
                    max_new_tokens=10,
                )
                .detach()
                .cpu()
            )
            batch["pred"] = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # print(batch["pred"])
            batch["pred"] = [
                pred.strip()[: len(batch["target"][i])] for i, pred in enumerate(batch["pred"])
            ]

            metric.add_batch(predictions=batch["pred"], references=batch["target"])
    return metric.compute()


all_results = {}
for subset in data_subsets:
    prompt = open(f"data/direct_bbh_prompts/{subset}.txt").read().split("-----")[1:]
    dataset = load_dataset("lukaemon/bbh", subset)
    print(f"evaluating {subset}", end=" ")
    results = evaluate_samples(dataset["test"], prompt=prompt)
    all_results[subset] = results
    print(f"\r{subset}: {results}")

print("all results:")
print(all_results)
print(data_subsets)
for subset in data_subsets:
    print(f"{all_results[subset]['exact_match']}", end="\t")
