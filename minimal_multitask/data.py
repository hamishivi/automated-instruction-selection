'''
General place for datasets. Handle conversion to my desired format.
'''
import json
from datasets import load_dataset, Dataset
from minimal_multitask.run_mmlu_eval import construct_prompts
from minimal_multitask.run_alpaca_eval import create_prompt_with_tulu_chat_format


class TestDataset:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def get_all_test_prompts(self, num_samples=500, seed=42):
        raise NotImplementedError

# turns sample into something we can feed into a model
def construct_test_sample(tokenizer, sample, max_length=1024):
    prompt, label = sample['prompts'], sample['labels']
    # note no space between prompt and label
    # have to put max length! sometimes gpt-4 will generate a lot of text
    inputs = tokenizer(prompt + label + tokenizer.eos_token, return_tensors='pt', max_length=max_length, truncation=True)
    input_len = len(tokenizer(prompt).input_ids)
    labels = inputs['input_ids'][0].clone()
    labels[:input_len] = -100
    return {
        'input_ids': inputs['input_ids'][0],
        'attention_mask': inputs['attention_mask'][0],
        'labels': labels
    }


class MMLU(TestDataset):
    def get_all_test_prompts(self, num_samples=500, seed=42):
        prompts, labels = construct_prompts(self.tokenizer, use_chat_format=True)
        test_dataset = Dataset.from_dict({'prompts': prompts, 'labels': labels})
        make_test_sample = lambda x: construct_test_sample(self.tokenizer, x, max_length=1024)
        test_dataset = test_dataset.map(make_test_sample, load_from_cache_file=False)
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        test_dataset = test_dataset.shuffle(seed=seed).select(range(num_samples))
        return test_dataset

class AlpacaEval(TestDataset):
    def get_all_test_prompts(self, num_samples=805, seed=42):
        data = json.load(open('data/alpaca_farm_gpt_4_output.json'))
        prompts, labels = [], []
        for example in data:
            prompt = example["instruction"].strip()
            messages = [{"role": "user", "content": prompt}]
            prompt = create_prompt_with_tulu_chat_format(messages, self.tokenizer, add_bos=False)
            prompts.append(prompt)
            labels.append(example['output'].strip())
        test_dataset = Dataset.from_dict({'prompts': prompts, 'labels': labels})
        test_dataset = test_dataset.map(construct_test_sample, load_from_cache_file=False)
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        test_dataset = test_dataset.shuffle(seed=seed).select(range(num_samples))
        return test_dataset

class SquadEval(TestDataset):
    def get_all_test_prompts(self, num_samples=500, seed=42):
        test_dataset = load_dataset('squad', split='validation')
        def convert_squad_sample(sample):
            prompt = sample['question']
            label = sample['answers']['text'][0]
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": label}]
            return {'prompt': create_prompt_with_tulu_chat_format(messages, add_bos=False), 'label': label}
        test_dataset = test_dataset.map(convert_squad_sample, remove_columns=['id', 'title', 'context', 'question', 'answers'])
        test_dataset = test_dataset.map(lambda x: construct_test_sample(self.tokenizer, x, max_length=1024))
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        test_dataset = test_dataset.shuffle(seed=seed).select(range(num_samples))
        return test_dataset

DATASETS = {
    'mmlu': MMLU,
    'alpacafarm': AlpacaEval,
    'squad': SquadEval
}
