import argparse
import json
import os

parser = argparse.ArgumentParser(description="Construct BBH results")
parser.add_argument("--input_folder", type=str, help="Input file")
parser.add_argument("--output_file", type=str, help="Output file")
args = parser.parse_args()

# do some
samples = []
for file in os.listdir(args.input_folder):
    if file.endswith(".jsonl"):
        prompt = open(f'data/direct_bbh_prompts/{file.replace(".jsonl", "")}.txt').read().split("-----")[1]
        with open(os.path.join(args.input_folder, file), "r") as f:
            for line in f:
                data = json.loads(line)
                samples.append(
                    {
                        "input": prompt.strip() + "\n\nQ: " + data["input"],
                        "target": data["target"],
                        "prediction": data["prediction"],
                        "was_correct": data["target"] == data["prediction"],
                    }
                )

with open(args.output_file, "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")
