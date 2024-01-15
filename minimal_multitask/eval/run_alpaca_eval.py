'''
Stripped-down version of the open-instruct evaluation script.
'''
import os
import json
import argparse
import logging
import random
import torch
import datasets
import vllm
from alpaca_eval import evaluate as alpaca_farm_evaluate
from transformers import AutoTokenizer


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
                "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
                )
    formatted_text += "<|assistant|>\n"
    formatted_text = bos + formatted_text if add_bos else formatted_text
    return formatted_text


def main(args):
    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    logging.info("loading data and model...")
    alpaca_eval_data = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    prompts = []
    chat_formatting_function = create_prompt_with_tulu_chat_format
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path if args.tokenizer_name_or_path is not None else args.model_name_or_path)

    for example in alpaca_eval_data:
        prompt = example["instruction"]
        messages = [{"role": "user", "content": prompt}]
        prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
        prompts.append(prompt)

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
    import pdb; pdb.set_trace()
    outputs = model.generate(prompts, sampling_params)
    outputs = [it.outputs[0].text for it in outputs]

    model_name = os.path.basename(os.path.normpath(args.model_name_or_path)) if args.model_name_or_path is not None else args.openai_engine
    model_results = []
    with open(os.path.join(args.save_dir, f"{model_name}-greedy-long-output.json"), "w") as fout:
        for example, output in zip(alpaca_eval_data, outputs):
            example["output"] = output
            example["generator"] = f"{model_name}-greedy-long"
            fout.write(json.dumps(example) + "\n")
            model_results.append(example)

    df_leaderboard, annotations = alpaca_farm_evaluate(
        model_outputs=model_results,
        annotators_config="alpaca_eval_gpt4_0314",
        output_path=args.save_dir,
        is_return_instead_of_print=True,
    )

    print(df_leaderboard.to_string(float_format="%.2f"))

    # save to json
    with open(os.path.join(args.save_dir, f"metrics.json"), "w") as fout:
        json.dump(df_leaderboard.to_dict(), fout)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str, 
        default="results/alpaca_farm")
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
    args = parser.parse_args()

    main(args)