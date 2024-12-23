import json
import random
from collections import defaultdict
import os

# Set the input and output file paths
INPUT_FILE = 'data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/tulu_v2_unfiltered_data_dedup.jsonl'
OUTPUT_DIR = 'data/source_exp'

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the science domains to be collapsed
SCIENCE_DOMAINS = [
    'science.evidence_inference',
    'science.qasper_truncated_4000',
    'science.scifact_json',
    'science.scitldr_aic',
    'science.scierc_ner',
    'science.scierc_relation'
]

# Function to read the JSONL file and group samples by source
def load_data(input_file):
    dataset_groups = defaultdict(list)
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            dataset_name = sample.get("dataset")
            if dataset_name:
                # Collapse the science datasets into a single 'science' domain
                if dataset_name in SCIENCE_DOMAINS:
                    dataset_name = 'science'
                dataset_groups[dataset_name].append(sample)
    return dataset_groups

# Function to balance samples across datasets and create 11 files
def create_balanced_files(dataset_groups, max_samples_per_file=100_000):
    dataset_names = list(dataset_groups.keys())
    total_datasets = len(dataset_names)

    for i in range(1, total_datasets + 1):
        print(f"Creating file with {i} datasets...")
        
        # Select the first `i` datasets to use for this file
        selected_datasets = dataset_names[:i]
        
        # Initialize a list to store the samples for this file
        file_samples = []
        remaining_samples = max_samples_per_file
        
        # Calculate the maximum samples we can take from each dataset, adjusted for available samples
        dataset_sizes = {dataset: len(dataset_groups[dataset]) for dataset in selected_datasets}
        
        # First, try to take an even number of samples from each dataset, capped by their available size
        while remaining_samples > 0 and selected_datasets:
            # If remaining_samples is less than the number of datasets, we will take 1 sample per dataset in a round-robin fashion
            if remaining_samples < len(selected_datasets):
                samples_per_dataset = 1
            else:
                samples_per_dataset = remaining_samples // len(selected_datasets)
            
            for dataset in selected_datasets[:]:  # Iterate over a copy of the list
                available_samples = len(dataset_groups[dataset])
                samples_to_take = min(available_samples, samples_per_dataset)
                
                if samples_to_take > 0:
                    # Randomly sample from the dataset
                    file_samples.extend(random.sample(dataset_groups[dataset], samples_to_take))
                    dataset_groups[dataset] = dataset_groups[dataset][samples_to_take:]  # Remove selected samples
                    remaining_samples -= samples_to_take
                
                # If a dataset is exhausted, remove it from the list
                if len(dataset_groups[dataset]) == 0:
                    selected_datasets.remove(dataset)
        
        # Check if we successfully filled the 100k quota
        if remaining_samples > 0 and len(file_samples) < max_samples_per_file:
            print(f"Warning: File has {len(file_samples)} samples, fewer than the target of {max_samples_per_file}.")
        
        # Write the selected samples to the output file
        output_file = os.path.join(OUTPUT_DIR, f"dataset_{i}_sources.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in file_samples:
                f.write(json.dumps(sample) + '\n')
        
        print(f"File '{output_file}' created with {len(file_samples)} samples.")

# Main function to execute the script
def main():
    # Step 1: Load the dataset samples from the input file
    dataset_groups = load_data(INPUT_FILE)
    
    # Step 2: Create 11 output files with exactly 100k samples per file
    create_balanced_files(dataset_groups)

if __name__ == "__main__":
    main()