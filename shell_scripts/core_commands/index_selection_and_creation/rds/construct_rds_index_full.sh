set -ex

BEAKER_PATH=01J8GPRF45KQNY6Y1HJWREVN2S

GANTRY_CMD="gantry run --cluster ai2/allennlp-cirrascale --cluster ai2/pluto-cirrascale --budget ai2/oe-adapt --allow-dirty --priority normal --preemptible --workspace ai2/minimal-multitask-finetuning --gpus 1 --env-secret OPENAI_API_KEY=OPENAI_API_KEY --env LD_LIBRARY_PATH=/opt/conda/envs/venv/lib --env IS_ALPACA_EVAL_2=False --dataset ${BEAKER_PATH}:/model --dataset 01J6QSXVDS4MN0W45HB2MHWXQN:/data"

# for file in /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/subshards/*.jsonl; do
#     shard=$(basename $file .jsonl)
#     echo "Processing $shard"
#     $GANTRY_CMD --name rds_sharded_full_create_${shard} --task-name rds_sharded_full_create_${shard} -- \
#         python -m minimal_multitask.compute_influence_cosinesim \
#             --model_name /model \
#             --seed 42 \
#             --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/subshards/${shard}.jsonl \
#             --eval_dataset alpacafarm \
#             --index_path /results/cosine_train_reps.pt \
#             --save_dir /results \
#             --batch_size 1
# done

# alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots
# for dataset in wildchat arena_hard; do
#     # replace file with beaker ids 
#     shards_dataset_file=index_lists/rds_fullscale.txt
#     while IFS=$'\t' read -r shard dataset_id; do
#             echo "Processing $shard $dataset_id"
#             $GANTRY_CMD --name rds_full_dedup_size_${dataset}_${shard} --task-name rds_full_dedup_size_${dataset}_${shard} --dataset "${dataset_id}:/index" -- python -m minimal_multitask.compute_influence_cosinesim \
#                     --model_name /model \
#                     --seed 42 \
#                     --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/subshards/tulu_v2_unfiltered_data_dedup_${shard}.jsonl \
#                     --eval_dataset $dataset \
#                     --index_path /index/cosine_train_reps.pt \
#                     --save_dir /results \
#                     --batch_size 1
#     done < "$shards_dataset_file"
# done

# for dataset in ; do
#     python -m minimal_multitask.compute_influence_cosinesim \
#         --model_name model_random_200k_to_10k \
#         --seed 42 \
#         --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/tulu_v2_unfiltered_data_dedup.jsonl \
#         --eval_dataset $dataset \
#         --index_path ${folder_name}/cosine_train_reps.pt \
#         --save_dir ${folder_name}/ \
#         --batch_size 1
# done


mkdir -p selection_files/rds_full_dedup_size
# alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots wildchat
for dataset in arena_hard; do
    python minimal_multitask/get_shard_file.py --prefix rds_full_dedup_size_${dataset} --outfile index_lists/rds_full_dedup_size_${dataset}.txt

    mkdir -p selection_files/rds_full_dedup_size/${dataset}
    cd selection_files/rds_full_dedup_size/${dataset}
    train_datasets=()
    while IFS=$'\t' read -r name id; do
        # Call the beaker command with the ID, with an optional prefix
        echo "Fetching $name with ID $id"
        beaker dataset fetch "$id"
        mv ${dataset}_cossim.pkl ${dataset}_${name}_cossim.pkl
        train_datasets+=($name)
    done < ../../../index_lists/rds_full_dedup_size_${dataset}.txt
    cd ../../..

    train_dataset_prefix=/net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/subshards/tulu_v2_unfiltered_data_dedup_

    train_datasets_combined=()
    for ds_name in "${train_datasets[@]}"; do
    train_datasets_combined+=(${train_dataset_prefix}${ds_name}.jsonl)
    done
    python -m minimal_multitask.get_top_influences \
        --input_files selection_files/rds_full_dedup_size/${dataset}/*.pkl \
        --output_file selection_files/rds_full_dedup_size/${dataset}/${dataset}_top10k.json \
        --output_size 10000 --selection_method max \
        --train_datasets ${train_datasets_combined[@]} \
        --output_dataset
done