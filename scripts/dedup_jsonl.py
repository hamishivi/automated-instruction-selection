'''
Quick script for deduplicating identical samples from jsonl files.
'''
import argparse
import json
from tqdm import tqdm


def dedup_jsonl(input_file, output_file):
    seen = set()
    count = 0
    with open(output_file, 'w') as out:
        for line in tqdm(open(input_file)):
            count += 1
            data = json.loads(line)
            # extract messages content and fill in.
            # just string to make it hashable
            data_content = "\n".join([d['role'] + ': ' + d['content'].strip() for d in data['messages']])
            if data_content not in seen:
                out.write(line)
                seen.add(data_content)
    return count - len(seen)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='Input jsonl file')
    parser.add_argument('output_file', type=str, help='Output jsonl file')
    args = parser.parse_args()
    removed = dedup_jsonl(args.input_file, args.output_file)
    print(f'Removed {removed} duplicates from {args.input_file} and saved to {args.output_file}')
