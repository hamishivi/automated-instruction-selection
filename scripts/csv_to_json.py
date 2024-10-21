'''
convert wildchat csv to json
'''
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Convert a CSV file to JSON format.')
parser.add_argument('csv_file', type=str, help='Path to the CSV file')
parser.add_argument('--output_file', type=str, default='output.json', help='Path to save the JSON file')
args = parser.parse_args()

# Load the CSV file
data = pd.read_csv(args.csv_file)
# we only want prompt, response.
data = data[['prompt', 'response']]
# Convert the DataFrame to a JSONl file
data.to_json(args.output_file, orient='records', lines=True)