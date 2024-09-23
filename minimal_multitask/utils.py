import torch
import os
import pickle
import numpy as np


# convert, but just return text.
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


# needed for open-instruct: convert msg format.
def encode_with_messages_format(example, tokenizer, max_seq_length, include_response=True, response_only=False, only_first_two=False):
    """
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    # change: just take the first two prompts.
    if only_first_two:
        # if first role is system, we actually want to take the second and third message,
        # ignoring the first system message.
        if messages[0]["role"] == "system":
            messages = messages[1:3]
        else:
            messages = messages[:2]
    if response_only:
        msg = "<|assistant|>\n" + messages[1]["content"].strip()
        res = tokenizer(msg, return_tensors="pt", max_length=max_seq_length, truncation=True)
        return {
            "string": msg,
            "input_ids": res.input_ids.flatten(),
            "attention_mask": res.attention_mask.flatten(),
        }
    elif not include_response:
        messages = [messages[0]]

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(example_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
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
        "string": messages_so_far,
    }


# helper script for working out if we need to look at /data
# or nfs
def get_appropriate_data_dir():
    # default to /data, in beaker.
    if os.path.exists("/net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data"):
        return "/net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data"
    elif os.path.exists("/data"):
        return "/data"
    elif os.path.exists("data"):
        return "data"
    else:
        raise FileNotFoundError("No valid data directory found.")


# a simple in-memory index for testing purposes.
class InMemoryFaiss:
    def __init__(self):
        self.is_trained = False
        self.vectors = []

    def add(self, vectors):
        self.vectors.extend(vectors)

    def search(self, query, k):
        # compute inner product between query and all vectors
        query = query.reshape(-1, 1)
        scores = np.stack(self.vectors, axis=0) @ query
        scores = scores[:, 0]
        # sort by score
        sorted_scores = np.argsort(scores)
        # return raw numbers and indices
        return scores[sorted_scores[:k]][None,], sorted_scores[:k][None,]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.vectors, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.vectors = pickle.load(f)
        self.is_trained = True
