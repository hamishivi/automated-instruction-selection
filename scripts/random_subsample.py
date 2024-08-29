import argparse
import random
import json

from tqdm import tqdm
from transformers import AutoTokenizer

from minimal_multitask.utils import encode_with_messages_format

parser = argparse.ArgumentParser(description='Randomly subsample a file')
parser.add_argument('input_file', type=str, help='Input file')
parser.add_argument('output_file', type=str, help='Output file')
parser.add_argument('n', type=int, help='Number of lines to sample')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

data = [json.loads(line) for line in tqdm(open(args.input_file))]
sampled_data = []

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# construct random indices
random_indices = list(range(len(data)))
random.seed(args.seed)
random.shuffle(random_indices)

# randomly sample n lines
# skip lines that are too long (no label after 2k)
pbar = tqdm(total=args.n)
while len(sampled_data) < args.n:
    line = data[random_indices.pop()]
    encoded_line = encode_with_messages_format(line, tokenizer, 1024, True, False)
    if (encoded_line['labels'] == -100).all():
        continue
    sampled_data.append(line)
    pbar.update(1)

# write sampled data to file
with open(args.output_file, 'w') as f:
    for line in sampled_data:
        f.write(json.dumps(line) + '\n')
