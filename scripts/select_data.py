import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Select a subset of samples from a data file.')
parser.add_argument('data_file', type=str, help='Path to the data file')
parser.add_argument('--output_file', type=str, default='selected_samples.csv', help='Path to save the selected samples')
args = parser.parse_args()

# Load your data file (change file path as needed)
data = pd.read_csv(args.data_file)

# Filter out unwanted labels
filtered_data = data[~data['predicted_label'].isin(['assisting or creative writing', 'other', 'recommendation'])]

# Initialize an empty list to store selected samples
selected_samples = []

# Number of samples required from each label
total_samples = 100
num_labels = len(filtered_data['predicted_label'].unique())
samples_per_label = total_samples // num_labels

# Keep track of the count of samples selected per label
label_counts = {label: 0 for label in filtered_data['predicted_label'].unique()}


# Define a function to display a prompt and get user decision
def show_example_and_select(row):
    print(f"\nPrompt: {row['prompt']}")
    print(f"Response: {row['response']}")
    decision = input("Do you want to add this example to the selected set? (y/n): ").strip().lower()
    return decision == 'y'


# Iterate through the data and show examples
for idx, row in filtered_data.iterrows():
    label = row['predicted_label']

    # Skip if we've already selected enough samples for this label
    if label_counts[label] >= samples_per_label:
        continue

    # Show the example and ask for user input
    if show_example_and_select(row):
        selected_samples.append(row)
        label_counts[label] += 1

    # Stop once we've selected enough samples for all labels
    if sum(label_counts.values()) >= total_samples:
        break

# Convert the selected samples to a DataFrame
selected_samples_df = pd.DataFrame(selected_samples)

# Save the selected samples to a new CSV file (optional)
selected_samples_df.to_csv(args.output_file, index=False)

print(f"\nYou have selected {len(selected_samples)} samples in total.")
