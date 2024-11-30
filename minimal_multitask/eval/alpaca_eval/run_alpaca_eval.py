"""
Stripped-down version of the open-instruct evaluation script.
"""
import os
import json
import argparse
import logging
import random
import tqdm

import torch
from datasets import Dataset, load_dataset
try:
    import vllm
except ImportError:
    print("VLLM not installed. Will not be able to use VLLM.")
    vllm = None
from alpaca_eval.main import evaluate as alpaca_farm_evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM
from minimal_multitask.eval.utils import dynamic_import_function

from torch.utils.data import DataLoader


def create_prompt_with_tulu_chat_format(messages, tokenizer, add_bos=False):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError(
                "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                    message["role"]
                )
            )
    formatted_text += "<|assistant|>\n"
    return formatted_text


def main(args):
    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    logging.info("loading data and model...")
    # original alpaca eval data
    # alpaca_eval_data = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    # my eval split :)
    alpaca_eval_data = load_dataset("hamishivi/alpaca_eval_test_mmft", split="test")

    chat_formatting_function = dynamic_import_function(args.chat_formatting_function)

    prompts = []
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    for example in alpaca_eval_data:
        prompt = example["instruction"]
        messages = [{"role": "user", "content": prompt}]
        prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
        prompts.append(prompt)
    print("Length of the datasets: ", len(prompts))

    model_name = (
        os.path.basename(os.path.normpath(args.model_name_or_path))
        if args.model_name_or_path is not None
        else args.openai_engine
    )
    model_results = []
    result_file = os.path.join(args.save_dir, f"{model_name}-greedy-long-output.jsonl")
    if args.results_file is not None:
        result_file = args.results_file
    if not os.path.exists(result_file):
        if args.use_vllm:
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path
                if args.tokenizer_name_or_path is not None
                else args.model_name_or_path,
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
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
            )
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.unk_token
            prompts = tokenizer(prompts, return_tensors="pt", padding=True)
            ds = Dataset.from_dict(prompts)
            ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
            ds_loader = DataLoader(ds, batch_size=args.batch_size)

            model.generation_config.max_length = 2000
            # Temporariy fix to stop when generating ".</", although it's weird that the model generate this instead of eos token
            model.generation_config.eos_token_id = [2, 21106, 829]
            if args.decoding_algo == "greedy":
                model.generation_config.temperature = 0.0
            elif args.decoding_algo == "sampling":
                model.generation_config.temperature = args.temperature
            outputs = []
            for batch in tqdm.tqdm(ds_loader):
                output = model.generate(batch["input_ids"].cuda(), attention_mask=batch["attention_mask"].cuda())
                generation = output[:, batch["input_ids"].shape[1]:]
                outputs.extend(generation)
            generations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, "w") as fout:
            for example, output in zip(alpaca_eval_data, generations):
                if output.endswith("</"):
                    output = (output.strip())[:-2]
                example["output"] = output.strip()  # get the rid of the </ at the end
                example["generator"] = f"{model_name}-greedy-long"
                fout.write(json.dumps(example) + "\n")
                model_results.append(example)
    else:
        print("Loading from existing file: ", result_file)
        with open(result_file, "r") as fout:
            for i in fout:
                model_results.append(json.loads(i))

    df_leaderboard, annotations = alpaca_farm_evaluate(
        model_outputs=model_results,
        annotators_config="alpaca_eval_gpt4",
        output_path=args.save_dir,
        is_return_instead_of_print=True,
        is_overwrite_leaderboard=True,
    )

    print(df_leaderboard.to_string(float_format="%.2f"))

    # save to json
    results_json = {"win_rate": df_leaderboard.to_dict()["win_rate"]["model-greedy-long"]}
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump(results_json, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="results/alpaca_farm")
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
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--results_file", type=str)
    parser.add_argument("--decoding_algo", type=str, default="greedy", choices=["greedy", "sampling"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
         default="minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`.",
    )
    args = parser.parse_args()

    main(args)
