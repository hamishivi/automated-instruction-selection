import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--input_requests", type=str)
parser.add_argument("--input_predictions", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--metric_name", type=str)
args = parser.parse_args()

samples = []
with open(args.input_requests, "r", encoding='utf-8') as f:
    for line in f:
        request = json.loads(line)
        if "messages" not in request["request"]["context"]:
            messages = [{"role": "user", "content": request["request"]["context"]}]
        else:
            messages = request["request"]["context"]["messages"]
        sample_id = f'{request["task_name"]}_{request["doc_id"]}_{request["native_id"]}'
        samples.append({
            "id": sample_id,
            "messages": messages
        })

with open(args.input_predictions, "r", encoding='utf-8') as f:
    for i, line in enumerate(f):
        predictions = json.loads(line)
        score = predictions["metrics"][args.metric_name]
        samples[i]["score"] = score

# output the samples
with open(args.output_file, "w", encoding='utf-8') as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")
