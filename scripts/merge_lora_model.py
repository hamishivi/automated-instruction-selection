'''
Quick script for merging loras into underlying model
'''
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel 

parser = argparse.ArgumentParser(description='Merge lora into underlying model')
parser.add_argument('model', type=str, help='Path to model')
parser.add_argument('lora', type=str, help='Path to lora')
parser.add_argument('output', type=str, help='Path to output')
parser.add_argument('--add_padding_token', action='store_true', help='Add padding token to tokenizer')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.lora)
base_model = AutoModelForCausalLM.from_pretrained(args.model)
if args.add_padding_token:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    base_model.resize_token_embeddings(len(base_model.tokenizer))
model = PeftModel.from_pretrained(base_model, args.lora)


model = model.merge_and_unload()
model.save_pretrained(args.output)
tokenizer.save_pretrained(args.output)
