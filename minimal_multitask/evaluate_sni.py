import re
import random
import errno
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from sni.sni_collator import DataCollatorForNI
import json
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import argparse

from sni.sni_evaluation import compute_all_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="google/t5-xl-lm-adapt")
parser.add_argument("--tokenizer", type=str, default="t5-base")
parser.add_argument("--output_folder", type=str, default='/output')
args = parser.parse_args()


model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
model.cuda()

random_gen = random.Random(42)


data_collator = DataCollatorForNI(tokenizer, num_pos_examples=0, text_only=True, max_source_length=768, max_target_length=256)
def convert_format(example):
        task_idx = re.findall(r"^task(\d+)_", example["Task"])
        assert len(task_idx) == 1
        task_idx = int(task_idx[0])
        processed_res = data_collator(example)
        return {
            "id": example["id"],
            "targets": random_gen.choice(example["Instance"]["output"]),
            "references": example["Instance"]["output"],
            "input": processed_res["inputs"][0]
            #**res
        }

ds = load_dataset("minimal_multitask/sni/sni_dataset.py")['test']
original_columns = ds.column_names
ds = ds.map(convert_format)
ds.set_format("pt")

ins = []
ids = []
for sample in ds:
    ins.append(sample['input'])
    ids.append(sample['id'])

outs = []
print('generating...')
with torch.inference_mode():
    for inputs in tqdm(KeyDataset(ds, "input"), total=len(ins)):
        inputs = tokenizer(inputs, return_tensors='pt').to(0)
        len_ids = inputs.input_ids.shape[-1]
        output_tokens = model.generate(**inputs, do_sample=False, max_length=1024)[0]
        output = tokenizer.decode(output_tokens[len_ids:], skip_special_tokens=True)
        outs.append(output)

try:
    os.mkdir(args.output_folder)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

predictions = {}
for i, o in zip(ids, outs):
    predictions[i] = o

with open(os.path.join(args.output_folder, 'predictions.json'), 'w') as w:
    w.write(json.dumps(predictions))

eval_instances = {}
with open('minimal_multitask/sni/test_references.jsonl') as fin:
    for line in fin:
        instance = json.loads(line)
        # if track is not provided in the refernce file, we use set the track
        # to `default` and use the default tokenizer in rouge-score.
        if "track" not in instance:
            instance["track"] = "default"
        eval_instances[instance["id"]] = instance

compute_all_metrics(predictions, eval_instances, os.path.join(args.output_folder, 'metrics.json'))
