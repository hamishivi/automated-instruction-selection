"""
Constructing sentence embeddings from llama
support just using eos token rep, or using sgpt weighted mean (https://arxiv.org/abs/2202.08904)
"""

import argparse
import json
import pickle
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# needed for open-instruct: convert msg format.
def encode_with_messages_format(example, tokenizer, max_seq_length, include_reponse=True):
    """
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    # change: just take the first two prompts.
    messages = messages[:2]
    # we may have prompt-only stuff.
    if not include_reponse:
        messages = [messages[0]]

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += (
                    "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
                )
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]),
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[: message_idx + 1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[: message_idx + 1])
            message_end_idx = tokenizer(
                messages_so_far, return_tensors="pt", max_length=max_seq_length, truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def main(args):
    assert args.use_eos or args.use_sgpt, "Must use either eos or sgpt methods for encoding."

    model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().cuda().half()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    camel_datasets = [
        "baize",
        "code_alpaca",
        "cot",
        "dolly",
        "flan_v2",
        "gpt4_alpaca",
        "oasst1",
        "self_instruct",
        "sharegpt",
        "stanford_alpaca",
        "super_ni",
        "unnatural_instructions",
    ]

    path = "/net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/camel_datasets"

    camels = {}
    camel_lengths = {}
    for filename in camel_datasets:
        with open(f"{path}/{filename}/{filename}_data.jsonl", "r") as f:
            camels[filename] = [json.loads(x.strip()) for x in f]
            camel_lengths[filename] = len(camels[filename])

    if args.subsample:
        for filename in camels:
            random.Random(42).shuffle(camels[filename])
            camels[filename] = camels[filename][:1000]
            camel_lengths[filename] = 1000

    camel_encoded_data = []
    with torch.inference_mode():
        for f in tqdm(camels):
            for sample in tqdm(camels[f]):
                input_ids = encode_with_messages_format(
                    sample, tokenizer, 2048, args.include_reponse
                )["input_ids"][
                    None,
                ]
                if args.use_eos:
                    input_ids = torch.cat(
                        [input_ids, torch.ones((input_ids.size(0), 1)) * tokenizer.eos_token_id],
                        axis=-1,
                    )
                    encoded = model(input_ids.long().cuda(), output_hidden_states=True)
                    camel_encoded_data.append(encoded.hidden_states[-1][0, -1].detach().cpu())
                elif args.use_sgpt:
                    position_weights = (
                        torch.arange(input_ids.shape[1]) / torch.arange(input_ids.shape[1]).sum()
                    )
                    encoded = model(input_ids.long().cuda(), output_hidden_states=True)
                    sentence_embedding = (
                        position_weights[:, None].cuda() * encoded.hidden_states[-1][0]
                    ).sum(0)
                    camel_encoded_data.append(sentence_embedding.detach().cpu())

    # save data to plug in elsewhere
    np.save(args.save_name, torch.stack(camel_encoded_data).numpy())
    pickle.dump(
        {"camels": camels, "camel_lengths": camel_lengths},
        open(f"{args.save_name}_metadata.pkl", "wb"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
    )
    parser.add_argument("--subsample", action="store_true")
    parser.add_argument("--save_name", type=str, default="camel_encodings")
    parser.add_argument("--use_eos", action="store_true")
    parser.add_argument("--use_sgpt", action="store_true")
    parser.add_argument("--include_reponse", action="store_true")
    args = parser.parse_args()

    main(args)
