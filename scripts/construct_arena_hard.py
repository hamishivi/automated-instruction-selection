'''
Construct the arena hard selection files.
'''
import os
import pandas as pd
import argparse
import urllib.request


parser = argparse.ArgumentParser(description='Construct the arena hard selection files.')
parser.add_argument('output_file', type=str, default='arena_hard', help='Directory to save the selected samples')
args = parser.parse_args()

# download https://raw.githubusercontent.com/lm-sys/arena-hard-auto/main/data/arena-hard-v0.1/question.jsonl
urllib.request.urlretrieve("https://raw.githubusercontent.com/lm-sys/arena-hard-auto/main/data/arena-hard-v0.1/question.jsonl", "question.jsonl")
# download gpt-4 0613 answers
urllib.request.urlretrieve("https://raw.githubusercontent.com/lm-sys/arena-hard-auto/main/data/arena-hard-v0.1/model_answer/gpt-4-0613.jsonl", "gpt-4-0613.jsonl")

# load the data files
question_data = pd.read_json("question.jsonl", lines=True)
gpt4_data = pd.read_json("gpt-4-0613.jsonl", lines=True)

# merge the data files
merged_data = pd.merge(question_data, gpt4_data, on='question_id')
# turn into samples with prompts and label
final_data = []
for idx, row in merged_data.iterrows():
    prompt = row["turns"][0]["content"]
    label = row['choices'][0]['turns'][0]['content']
    final_data.append({'prompt': prompt, 'label': label})
# write to jsonl
final_data_df = pd.DataFrame(final_data)
final_data_df.to_json(args.output_file, orient='records', lines=True)

# remove the downloaded files
os.remove("question.jsonl")
os.remove("gpt-4-0613.jsonl")
