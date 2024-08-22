set -ex

mkdir -p $1
cd $1

cp ../shell_scripts/gantry_commands/beaker_ids_unfiltered_train.txt .

while read id; do
  beaker dataset fetch $id --prefix unfiltered_tulu_alpacafarm_samples_shard
done <beaker_ids_unfiltered_train.txt

train_datasets=(data/tulu_splits/tulu_v2_unfiltered/tulu_v2_unfiltered_data_shard1_1.jsonl data/tulu_splits/tulu_v2_unfiltered/tulu_v2_unfiltered_data_shard1_2.jsonl data/tulu_splits/tulu_v2_unfiltered/tulu_v2_unfiltered_data_shard1_3.jsonl data/tulu_splits/tulu_v2_unfiltered/tulu_v2_unfiltered_data_shard1_4.jsonl data/tulu_splits/tulu_v2_unfiltered/tulu_v2_unfiltered_data_shard1_5.jsonl data/tulu_splits/tulu_v2_unfiltered/tulu_v2_unfiltered_data_shard2.jsonl data/tulu_splits/tulu_v2_unfiltered/tulu_v2_unfiltered_data_shard3.jsonl data/tulu_splits/tulu_v2_unfiltered/tulu_v2_unfiltered_data_shard4.jsonl data/tulu_splits/tulu_v2_unfiltered/tulu_v2_unfiltered_data_shard5.jsonl data/tulu_splits/tulu_v2_unfiltered/tulu_v2_unfiltered_data_shard6.jsonl)

train_datasets=(data/tulu_splits/tulu_v2_unfiltered/tulu_v2_unfiltered_data_shard1.jsonl data/tulu_splits/tulu_v2_unfiltered/tulu_v2_unfiltered_data_shard2.jsonl data/tulu_splits/tulu_v2_unfiltered/tulu_v2_unfiltered_data_shard3.jsonl data/tulu_splits/tulu_v2_unfiltered/tulu_v2_unfiltered_data_shard4.jsonl data/tulu_splits/tulu_v2_unfiltered/tulu_v2_unfiltered_data_shard5.jsonl data/tulu_splits/tulu_v2_unfiltered/tulu_v2_unfiltered_data_shard6.jsonl)

cd ..

# next, we aggregate topk over them all!
python -m minimal_multitask.get_top_influences \
    --input_files $1/*.pkl \
    --output_file results/llama_2_128_lora_unfiltered_tulu/mmlu_unfiltered_topk.json \
    --output_size 10000 \
    --train_datasets ${train_datasets[@]} \
    --output_dataset  # we want to save the actual data

