set -ex


GANTRY_CMD="gantry run --cluster ai2/saturn-cirrascale --cluster ai2/neptune-cirrascale --cluster ai2/jupiter-cirrascale-2 --budget ai2/oe-adapt --allow-dirty --priority normal --preemptible --workspace ai2/minimal-multitask-finetuning --gpus 1 --env-secret HF_TOKEN=HF_TOKEN --env-secret OPENAI_API_KEY=OPENAI_API_KEY --env LD_LIBRARY_PATH=/opt/conda/envs/venv/lib --env IS_ALPACA_EVAL_2=False --dataset 01J6QSXVDS4MN0W45HB2MHWXQN:/data --weka oe-adapt-default:/weka"

while IFS=$'\n' read -r shard; do
    echo "Processing $shard"
    $GANTRY_CMD --name rds_sharded_tulu_3_full_create_${shard} --task-name rds_sharded_tulu_3_full_create_${shard} -- \
        python -m minimal_multitask.compute_influence_cosinesim \
            --model_name meta-llama/Llama-3.1-8B \
            --seed 42 \
            --train_dataset /weka/hamishi/influence-tuning/data/tulu_3.9/shards/part_${shard} \
            --eval_dataset alpacafarm \
            --index_path /results/cosine_train_reps.pt \
            --save_dir /results \
            --batch_size 1 \
            --pooling weighted_mean
done < shard_names.txt


for dataset in mmlu_shots wildchat arena_hard; do
    # replace file with beaker ids 
    mkdir -p rds_sels/${dataset}
    shards_dataset_file=rds_sels/beaker.txt
    while IFS=$'\t' read -r shard dataset_id; do
            echo "Processing $shard $dataset_id"
            python -m minimal_multitask.compute_influence_cosinesim \
                    --model_name meta-llama/Llama-3.1-8B \
                    --seed 42 \
                    --train_dataset data/tulu_3.9/shards/part_${shard} \
                    --eval_dataset $dataset \
                    --index_path rds_sels/${shard}/cosine_train_reps.pt \
                    --save_dir rds_sels/${shard}/${dataset} \
                    --batch_size 1 \
                    --pooling weighted_mean
    done < "$shards_dataset_file"
done

for dataset in alpacafarm bbh_shots codex gsm8k_shots tydiqa_shots mmlu_shots squad arena_hard wildchat; do
    python scripts/combine_pickles.py --input_files /weka/hamishi/influence-tuning/rds_sels/**/${dataset}/*.pkl --output_file /weka/hamishi/influence-tuning/rds_sels/${dataset}_cossim.pkl
done

for dataset in alpacafarm bbh_shots codex gsm8k_shots tydiqa_shots mmlu_shots squad arena_hard wildchat; do
    python -m minimal_multitask.get_top_optimized \
        --input_files /weka/hamishi/influence-tuning/rds_sels/${dataset}_cossim.json \
        --output_file /weka/hamishi/influence-tuning/rds_sels/${dataset}_top10k.json \
        --output_size 10000 --selection_method max \
        --train_datasets /weka/hamishi/influence-tuning/data/tulu_3.9/tulu_3_v3.9_unfiltered.jsonl

    python -m minimal_multitask.get_top_optimized \
        --input_files /weka/hamishi/influence-tuning/rds_sels/${dataset}_cossim.json \
        --output_file /weka/hamishi/influence-tuning/rds_sels/${dataset}_top320k.json \
        --output_size 320000 --selection_method max \
        --train_datasets /weka/hamishi/influence-tuning/data/tulu_3.9/tulu_3_v3.9_unfiltered.jsonl

    python -m minimal_multitask.get_top_optimized \
        --input_files /weka/hamishi/influence-tuning/rds_sels/${dataset}_cossim.json \
        --output_file /weka/hamishi/influence-tuning/rds_sels/${dataset}_top50k.json \
        --output_size 50000 --selection_method max \
        --train_datasets /weka/hamishi/influence-tuning/data/tulu_3.9/tulu_3_v3.9_unfiltered.jsonl
done

# finally round robin max
python -m minimal_multitask.get_top_aggregated_influences \
    --input_files /weka/hamishi/influence-tuning/rds_sels/*_cossim.json \
    --output_file /weka/hamishi/influence-tuning/rds_sels/multitask_rrmax_320k.json \
    --output_size 320000 --selection_method max \
    --train_dataset /weka/hamishi/influence-tuning/data/tulu_3.9/tulu_3_v3.9_unfiltered.jsonl \
    --output_dataset

# match size of tulu unfiltered mix.
python -m minimal_multitask.get_top_aggregated_influences \
    --input_files /weka/hamishi/influence-tuning/rds_sels/*_cossim.json \
    --output_file /weka/hamishi/influence-tuning/rds_sels/multitask_rrmax_939k.json \
    --output_size 939000 --selection_method max \
    --train_dataset /weka/hamishi/influence-tuning/data/tulu_3.9/tulu_3_v3.9_unfiltered.jsonl \
    --output_dataset

# arena hard large selection
python -m minimal_multitask.get_top_optimized \
    --input_files /data/input/hamishi/influence-tuning/rds_sels/arena_hard_cossim_combined.json \
    --output_file /data/input/hamishi/influence-tuning/rds_sels/arena_hard_top939k.json \
    --output_size 939000 --selection_method max \
    --train_datasets /data/input/hamishi/influence-tuning/data/tulu_3.9/tulu_3_v3.9_unfiltered.jsonl \
    --output_dataset

