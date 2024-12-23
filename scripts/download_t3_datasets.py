'''
Helper script to download tulu3 datasets we have been looking at.
'''
import json
from datasets import load_dataset
from argparse import ArgumentParser
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--output_file', type=str, default='data/tulu_3_data/tulu_3.jsonl')
args = parser.parse_args()

dataset_list = [
    "HuggingFaceH4/no_robots",
    "allenai/tulu-v2-sft-mixture",
    "ai2-adapt-dev/llama-3-tulu-v2-sft-mixture-with-subset-llama-405b-completions-code_alpaca-open_orca-gpt4_alpaca",
    "HuggingFaceH4/no_robots",
    "allenai/openassistant-guanaco-reformatted",
    "natolambert/tulu-v2-sft-mixture-lima",
    "ai2-adapt-dev/aya_dataset-reformat",
    "allenai/SciRIFF-train-mix",
    "natolambert/tulu-v2-sft-mixture-cot",
    "ai2-adapt-dev/SlimOrca-reformat",
    "ai2-adapt-dev/WildChat-1M-Full-GPT4-Only",
    "ai2-adapt-dev/WizardLM_evol_instruct_V2_196k_reformat",
    "Vtuber-plan/ShareGPT-cleaned",
    "ai2-adapt-dev/Daring-Anteater-reformat",
    "ai2-adapt-dev/metamath-qa-reformat",
    "ai2-adapt-dev/WebInstructSub-reformat",
    "m-a-p/CodeFeedback-Filtered-Instruction",
    "m-a-p/Code-Feedback",
    "ai2-adapt-dev/Table-GPT-All-train",
    "ai2-adapt-dev/coconot-sft-reformat",
    "AI-MO/NuminaMath-TIR",
    "Salesforce/xlam-function-calling-60k",
    "lmsys/lmsys-chat-1m"
]

seen_samples = set()

with open(args.output_file, 'w') as f:
    for dataset in tqdm(dataset_list, position=0):
        ds = load_dataset(dataset, split='train')
        for example in tqdm(ds, position=1):
            # some quick sanity checks
            try:
                assert 'messages' in example
                concat_msg = ''
                for message in example['messages']:
                    assert message['content']
                    concat_msg += message['content']
                if concat_msg in seen_samples:
                    continue
                seen_samples.add(concat_msg)
                f.write(json.dumps(example) + '\n')
            except Exception:
                continue
