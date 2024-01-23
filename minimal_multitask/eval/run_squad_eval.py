import torch
from datasets import load_dataset
import evaluate
from argparse import ArgumentParser
import vllm
from transformers import AutoTokenizer
from minimal_multitask.eval.run_alpaca_eval import create_prompt_with_tulu_chat_format

parser = ArgumentParser()
parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-70m')
parser.add_argument('--tokenizer', type=str, default=None)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

squad_og = load_dataset('squad', split='validation')
squad_inf_split_og = squad_og.shuffle(seed=42).select(range(500))

# convert everything to tulu
def convert_squad_sample(sample):
    prompt = sample['question'] + " Provide a short answer only."
    label = sample['answers']['text'][0]
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": label}]
    return {'prompt': create_prompt_with_tulu_chat_format(messages, tokenizer, add_bos=False), 'label': label}

squad = squad_og.map(convert_squad_sample, load_from_cache_file=False)
squad.set_format(type='torch', columns=['prompt', 'label'])
prompts = squad['prompt']

squad_inf_split = squad_inf_split_og.map(convert_squad_sample, load_from_cache_file=False)
squad_inf_split.set_format(type='torch', columns=['prompt', 'label'])
inf_prompts = squad_inf_split['prompt']

# predict over dataset
model = vllm.LLM(
        model=args.model_name,
        tokenizer=args.tokenizer if args.tokenizer is not None else args.model_name,
        tensor_parallel_size=torch.cuda.device_count(),
    )
sampling_params = vllm.SamplingParams(
    temperature=0,  # greedy decoding
    max_tokens=50,  # I don't think anything should go over this?
)

outputs = model.generate(prompts, sampling_params)
outputs = [it.outputs[0].text for it in outputs]

inf_outputs = model.generate(inf_prompts, sampling_params)
inf_outputs = [it.outputs[0].text for it in inf_outputs]

# calculate squad f1
f1_scorer = evaluate.load("squad")
references = [{'id': sample['id'], 'answers': sample['answers']} for sample in squad_og]
outputs = [{'id': squad_og[i]['id'], 'prediction_text': output} for i, output in enumerate(outputs)]
results = f1_scorer.compute(references=references, predictions=outputs)
inf_references = [{'id': sample['id'], 'answers': sample['answers']} for sample in squad_inf_split_og]
inf_outputs = [{'id': squad_inf_split_og[i]['id'], 'prediction_text': output} for i, output in enumerate(inf_outputs)]
inf_results = f1_scorer.compute(references=inf_references, predictions=inf_outputs)

print("Results on all squad:")
print(results)
print("Results on 500 squad:")
print(inf_results)