'''
construct random split of open orca, flan, and 50/50
'''
import random
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default='data')
args = parser.parse_args()

data_file = args.data_file
data = []
with open(data_file, 'r') as f:
    for line in f:
        data.append(json.loads(line))

random.shuffle(data)

flan_data = [x for x in data if x['dataset'] == 'flan_v2']
orca_data = [x for x in data if x['dataset'] == 'open_orca']

# construct our three splits: 10k flan, 10k orca, 5k each
flan_split = random.sample(flan_data, 10000)
orca_split = random.sample(orca_data, 10000)
split = random.sample(flan_data, 5000) + random.sample(orca_data, 5000)

with open('data/flan_split_10k.json', 'w') as f:
    for line in flan_split:
        f.write(json.dumps(line) + '\n')

with open('data/orca_split_10k.json', 'w') as f:
    for line in orca_split:
        f.write(json.dumps(line) + '\n')

with open('data/split_flan_orca_5050_10k.json', 'w') as f:
    for line in split:
        f.write(json.dumps(line) + '\n')
