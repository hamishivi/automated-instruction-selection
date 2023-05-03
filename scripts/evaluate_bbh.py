'''
Evaluate a hf model on BBH (a subset of big bench)
'''
import torch
import argparse
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Evaluate a hf model on BBH (a subset of big bench)')
parser.add_argument('--model_name', type=str, default='google/t5-xl-lm-adapt', help='model name')
parser.add_argument('--tokenizer_name', type=str, default=None, help='tokenizer name (defaults to model_name if not specified)')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--few_shot', action='store_true', help='evaluate on few shot (3-shot) setting to match Flan-T5 and BBH papers.')
args = parser.parse_args()

data_subsets = ['boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa', 'dyck_languages', 'formal_fallacies', 'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects', 'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding', 'temporal_sequences', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects', 'web_of_lies', 'word_sorting']

model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name)
metric = evaluate.load('exact_match')
# tokenize inputs

def evaluate(examples, prompt=None):
    with torch.inference_mode():
        dataloader = torch.utils.data.DataLoader(examples, batch_size=args.batch_size, shuffle=False)
        for batch in tqdm(dataloader):
            if prompt:
                inputs = [prompt + f'\n\nQ:{inp}\nA: ' for inp in batch['input']]
            else:
                inputs = batch['input']
            inputs = tokenizer(inputs, return_tensors='pt', padding=True, truncation=False)
            # most bbh outputs are very short
            outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=10)
            batch['pred'] = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # hacky, but needed at least for causal judgements...
            batch['pred'] = [pred.replace('A: ', '') for pred in batch['pred']]
            metric.add_batch(predictions=batch['pred'], references=batch['target'])
    return metric.compute()

all_results = {}
for subset in data_subsets:
    prompt = open(f'data/direct_bbh_prompts/{subset}.txt').read().split('-----')[-1]
    dataset = load_dataset('lukaemon/bbh', subset)
    print(f'evaluating {subset}', end=' ')
    results = evaluate(dataset['test'], prompt=prompt if args.few_shot else None)
    all_results[subset] = results
    print(f"\r{subset}: {results}")

print('all results:')
print(all_results)
