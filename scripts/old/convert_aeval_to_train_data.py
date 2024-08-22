import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
args = parser.parse_args()

with open(args.input_file, "r") as f:
    data = json.load(f)

dataset = []
for item in data:
    dataset.append(
        {
            "messages": [
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["output"]},
            ],
            "dataset": "alpacaeval",
        }
    )

with open(args.output_file, "w") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")
