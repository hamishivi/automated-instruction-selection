# script for downloading and selecting from pickles
# point it to a text file with the dataset names and ids, it downloads
# and then loads them all up.

set -ex

mkdir -p $1
cd $1

filename=$2

file_basename=$(basename $filename)
cp ../../../$2 .

train_datasets=()
while IFS=$'\t' read -r name id; do
  # Call the beaker command with the ID, with an optional prefix
  echo "Fetching $name with ID $id"
  if [ -z "$3" ]; then
    beaker dataset fetch "$id"
  else
    beaker dataset fetch "$id" --prefix $3
  fi
  train_datasets+=($name)
done < $file_basename

train_dataset_prefix=/net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/subshards/tulu_v2_unfiltered_data_dedup_

train_datasets_combined=()
for ds_name in "${train_datasets[@]}"; do
  train_datasets_combined+=(${train_dataset_prefix}${ds_name}.jsonl)
done
cd ../../..

# next, we aggregate topk over them all!
python -m minimal_multitask.get_top_influences \
    --input_files $1/*.pkl \
    --output_file $1/top_10k.json \
    --output_size 10000 \
    --selection_method normalized_mean_min \
    --train_datasets ${train_datasets_combined[@]} \
    --output_dataset  # we want to save the actual data

