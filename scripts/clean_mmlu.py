import json
import sys

file = f"data/camel_results/bbh_cot_3shot/{sys.argv[1]}.jsonl"

with open(file, "r") as f:
    lines = f.readlines()

lines = [json.loads(line.strip()) for line in lines]

new_lines = []
for line in lines:
    samples = line["input"].split("\n\n")
    new_line = "\n\n".join([samples[0], samples[-1]])
    line["input"] = new_line
    new_lines.append(line)

with open(f"data/camel_results/bbh_cot_0shot/{sys.argv[1]}.jsonl", "w") as f:
    for line in new_lines:
        f.write(json.dumps(line) + "\n")
