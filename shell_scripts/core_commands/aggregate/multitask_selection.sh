# script for downloading and selecting from pickles
# point it to a text file with the dataset names and ids, it downloads
# and then loads them all up.

set -ex

mkdir -p $1

# next, we aggregate topk over them all!
python -m minimal_multitask.get_top_aggregated_influences \
    --input_files ../../hamishi/minimal-multitask-tuning/lora_ff_10k_fixed_sampling_combined/combined_*.json \
    --output_file $1.json \
    --output_size $2 \
    --selection_method $3 \
    --aggregation_method $4 \
    --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/tulu_v2_unfiltered_data_dedup.jsonl \
    --output_dataset  # we want to save the actual data

python3 -m minimal_multitask.get_top_influences --input_files /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/selection_files/wildchat_lora_ff_10k_fixed_sampling/combined.json --output_file selections/lora_ff_10k_fixed_sampling/wildchat_normalizedmean_326154.json --selection_method "normalized_mean_min" --train_datasets /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/tulu_v2_unfiltered_data_dedup.jsonl --output_dataset --output_size 326154