import argparse
import os
import tqdm
import json

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
import evaluate
import vllm
from transformers import AutoTokenizer, AutoModelForCausalLM
from minimal_multitask.eval.alpaca_eval.run_alpaca_eval import create_prompt_with_tulu_chat_format

def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

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
    print("Length of the datasets: ", len(prompts))

    # squad_inf_split = squad_inf_split_og.map(convert_squad_sample, load_from_cache_file=False)
    # squad_inf_split.set_format(type='torch', columns=['prompt', 'label'])
    # inf_prompts = squad_inf_split['prompt']

    # predict over dataset
    model_name = os.path.basename(os.path.normpath(args.model_name_or_path)) if args.model_name_or_path is not None else args.openai_engine
    model_results = []
    result_file = os.path.join(args.save_dir, f"{model_name}-greedy-long-output.jsonl")
    if args.results_file is not None:
        result_file = args.results_file
    if not os.path.exists(result_file):
        if args.use_vllm:
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path is not None else args.model_name_or_path,
                # tokenizer_mode="slow",
                tensor_parallel_size=torch.cuda.device_count(),
            )
            sampling_params = vllm.SamplingParams(
                temperature=0,  # greedy decoding
                max_tokens=8192,
            )
            outputs = model.generate(prompts, sampling_params)
            generations = [it.outputs[0].text for it in outputs]
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
            tokenizer.padding_side = 'left'
            tokenizer.pad_token = tokenizer.unk_token
            hf_prompts = tokenizer(prompts, return_tensors='pt', padding=True)
            ds = Dataset.from_dict(hf_prompts)
            ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
            ds_loader = DataLoader(ds, batch_size=args.batch_size)

            model.generation_config.max_new_tokens = 100
            # model.generation_config.eos_token_id = [2, 21106, 829] # Temporariy fix to stop when generating ".</", although it's weird that the model generate this instead of eos token
            if args.decoding_algo == "greedy":
                model.generation_config.temperature=0.0
                model.generation_config.do_sample=False
            elif args.decoding_algo == "sampling":
                model.generation_config.temperature=args.temperature
            outputs = []
            for batch in tqdm.tqdm(ds_loader):
                output = model.generate(batch['input_ids'].cuda(), attention_mask=batch['attention_mask'].cuda())
                generation = output[:, batch['input_ids'].shape[1]:]
                outputs.extend(generation)
            generations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, "w") as fout:
            for i, output in enumerate(generations):
                # if output.endswith("</"):
                #     output = (output.strip())[:-2]
                fout.write(json.dumps({"Prompt" : prompts[i], "Generation" : (output.strip())}) + "\n")
                model_results.append({'id': squad_og[i]['id'], 'prediction_text': output})
    else:
        print("Loading from existing file: ", result_file)
        with open(result_file, "r") as fout:
            for i, content in enumerate(fout):
                output = json.loads(content)['Generation']
                model_results.append({'id': squad_og[i]['id'], 'prediction_text': output})

    # inf_outputs = model.generate(inf_prompts, sampling_params)
    # inf_outputs = [it.outputs[0].text for it in inf_outputs]

    # calculate squad f1c
    f1_scorer = evaluate.load("squad")
    references = [{'id': sample['id'], 'answers': sample['answers']} for sample in squad_og]
    outputs = [{'id': squad_og[i]['id'], 'prediction_text': output} for i, output in enumerate(outputs)]
    results = f1_scorer.compute(references=references, predictions=model_results)
    inf_references = [{'id': sample['id'], 'answers': sample['answers']} for sample in squad_inf_split_og]
    inf_outputs = [{'id': squad_inf_split_og[i]['id'], 'prediction_text': output} for i, output in enumerate(inf_outputs)]
    inf_results = f1_scorer.compute(references=inf_references, predictions=inf_outputs)

    print("Results on all squad:")
    print(results)
    print("Results on 500 squad:")
    print(inf_results)

    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(f"Results on all squad:\n{results}\nResults on 500 squad:\n{inf_results}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument("--use_vllm", action='store_true')
    parser.add_argument("--results_file", type=str)
    parser.add_argument("--decoding_algo", type=str, default="greedy", choices=["greedy", "sampling"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    if not args.save_dir:
        args.save_dir = args.model_name_or_path

    main(args)