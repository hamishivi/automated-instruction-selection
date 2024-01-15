from minimal_multitask.eval.run_mmlu_eval import construct_prompts
from datasets import Dataset
import os
import torch
import csv
from statistics import mean
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, default='EleutherAI/pythia-70m')
args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')

# load and process test dataset
prompts, labels = construct_prompts(tokenizer)
def construct_mmlu_sample(sample):
    prompt, label = sample['prompts'], sample['labels']
    inputs = tokenizer(prompt + label, return_tensors='pt')
    input_len = len(tokenizer(prompt).input_ids)
    labels = torch.ones(inputs['input_ids'].shape[1], dtype=torch.long)
    labels[:input_len] = -100
    return {
        'input_ids': inputs['input_ids'][0],
        'attention_mask': inputs['attention_mask'][0],
        'labels': labels
    }
test_dataset = Dataset.from_dict({'prompts': prompts, 'labels': labels})
test_dataset = test_dataset.map(construct_mmlu_sample)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset = test_dataset.shuffle(seed=42).select(range(500))

sub_prompts = [tokenizer.decode(prompt) for prompt in test_dataset['input_ids']]

# load predictions
preds = []
for file in os.listdir(args.input_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(args.input_folder, file)
        with open(file_path, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            head = False
            for row in csv_reader:
                if not head:
                    head = True
                    continue
                if any([(row[0] in prompt and row[1] in prompt and row[2] in prompt and row[3] in prompt and row[4] in prompt) for prompt in sub_prompts]):
                    preds.append(row[-5])

print(len(preds))
print(mean([pred == 'True' for pred in preds if pred != 'correct']))
