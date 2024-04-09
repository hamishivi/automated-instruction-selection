import json
import torch
from datasets import load_dataset
import evaluate
from argparse import ArgumentParser
import vllm
from transformers import AutoTokenizer
from minimal_multitask.eval.alpaca_eval.run_alpaca_eval import create_prompt_with_tulu_chat_format
from minimal_multitask.eval.squad.squad_eval_1 import evaluate

parser = ArgumentParser()
parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-70m')
parser.add_argument('--tokenizer', type=str, default=None)
parser.add_argument('--output_file', type=str, default=None)
parser.add_argument('--metrics_file', type=str, default=None)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

squad_og = load_dataset('squad', split='validation')

# convert everything to tulu
def convert_squad_sample(sample):
    prompt = sample['context'] + '\n\n' + sample['question']
    label = sample['answers']['text'][0]
    messages = [{"role": "user", "content": prompt}]
    prompt = create_prompt_with_tulu_chat_format(messages, tokenizer, add_bos=False) # + 'Answer: '
    return {'prompt': prompt, 'label': label}

squad = squad_og.map(convert_squad_sample, load_from_cache_file=False)
squad.set_format(type='torch', columns=['prompt', 'label'])
prompts = squad['prompt']

# predict over dataset
model = vllm.LLM(
        model=args.model_name,
        tokenizer=args.tokenizer if args.tokenizer is not None else args.model_name,
        tensor_parallel_size=torch.cuda.device_count(),
    )
sampling_params = vllm.SamplingParams(
    temperature=0,  # greedy decoding
    max_tokens=2048,  # I don't think anything should go over this?
)

outputs = model.generate(prompts, sampling_params)
outputs = [it.outputs[0].text.strip() for it in outputs]

# calculate squad f1
references = [{'id': sample['id'], 'answers': sample['answers']} for sample in squad_og]
outputs = [{'id': squad_og[i]['id'], 'prediction_text': output} for i, output in enumerate(outputs)]
results = evaluate(references=references, predictions=outputs)
print("Results on all squad:")
print(results)

if args.output_file:
    with open(args.output_file, 'w') as f:
        for sample in outputs:
            f.write(json.dumps(sample) + '\n')
if args.metrics_file:
    results_dict = results
    with open(args.metrics_file, 'w') as f:
        f.write(json.dumps(results_dict))