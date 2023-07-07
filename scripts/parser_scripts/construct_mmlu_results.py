import argparse
import os
import json
import csv

parser = argparse.ArgumentParser(description='Construct BBH results')
parser.add_argument('--input_folder', type=str, help='Input folder')
parser.add_argument('--output_file', type=str, help='Output file')
args = parser.parse_args()

choices = ["A", "B", "C", "D"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(options, answer=None):
    k = len(options)
    prompt = ""
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], options[j])
    prompt += "\nAnswer:"
    if answer:
        prompt += " {}\n\n".format(answer)
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    for i in range(k):
        data = train_df[i]
        question = data[0]
        prompt += question
        options = [data[1], data[2], data[3], data[4]]
        prompt += format_example(options, answer=data[5])
    return prompt

# do some 
samples = []
for file in os.listdir(args.input_folder):
    if file.endswith(".csv"):
        subject = file.replace('.csv', '').replace('_', ' ')

        dev_data = csv.reader(open(f"/net/nfs.cirrascale/allennlp/yizhongw/open-instruct/data/eval/mmlu/dev/{file.replace('.csv', '')}_dev.csv"))
        dev_data = list(dev_data)
        
        few_shot_prompt = gen_prompt(dev_data, subject, k=5)

        with open(os.path.join(args.input_folder, file), 'r') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for i, data in enumerate(reader):
                if i == 0:
                    continue
                question = data[0]
                options = [data[1], data[2], data[3], data[4]]
                was_correct = data[6] == "True"
                model_probs = [data[7], data[8], data[9], data[10]]
                model_pred = choices[model_probs.index(max(model_probs))]



                prompt = few_shot_prompt + data[0] + format_example(options)

                samples.append({
                    "input": prompt.strip(),
                    "target": data[5],
                    "prediction": model_pred,
                    "was_correct": was_correct,
                })
                
with open(args.output_file, 'w') as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")
