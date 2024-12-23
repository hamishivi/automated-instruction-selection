
set -ex

BEAKER_PATH=01J8GPRF45KQNY6Y1HJWREVN2S

GANTRY_CMD="gantry run --cluster ai2/allennlp-elara-cirrascale   --cluster ai2/neptune-cirrascale  --cluster ai2/saturn-cirrascale --cluster ai2/jupiter-cirrascale-2 --weka=oe-adapt-default:/weka --budget ai2/oe-adapt --allow-dirty --priority normal --preemptible --workspace ai2/minimal-multitask-finetuning --gpus 1 --env-secret OPENAI_API_KEY=OPENAI_API_KEY --env LD_LIBRARY_PATH=/opt/conda/envs/venv/lib --env IS_ALPACA_EVAL_2=False --dataset 01J6QSXVDS4MN0W45HB2MHWXQN:/data"

# compute influence for each shard
while IFS=$'\n' read -r shard; do
    echo "Processing $shard"
    mkdir -p rds_sels_qwen/${shard}
    # $GANTRY_CMD --name rds_qwen15_sharded_full_create_${shard} --task-name rds_qwen15_sharded_full_create_${shard} -- \
    python -m minimal_multitask.compute_influence_cosinesim \
        --model_name Qwen/Qwen2.5-1.5B \
        --seed 42 \
        --train_dataset subshards/tulu_v2_unfiltered_data_dedup_${shard}.jsonl \
        --eval_dataset alpacafarm \
        --index_path rds_sels_qwen/${shard}/cosine_train_reps.pt \
        --save_dir rds_sels_qwen/${shard} \
        --batch_size 1 \
        --pooling weighted_mean
done < shard_names.txt

for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
    # replace file with beaker ids 
    mkdir -p rds_sels_qwen/${dataset}
    while IFS=$'\t' read -r shard; do
            echo "Processing $shard"
            python -m minimal_multitask.compute_influence_cosinesim \
                    --model_name Qwen/Qwen2.5-1.5B \
                    --seed 42 \
                    --train_dataset subshards/tulu_v2_unfiltered_data_dedup_${shard}.jsonl \
                    --eval_dataset $dataset \
                    --index_path rds_sels_qwen/${shard}/cosine_train_reps.pt \
                    --save_dir rds_sels_qwen/${shard}/${dataset} \
                    --batch_size 1 \
                    --pooling weighted_mean
    done < shard_names.txt
done

for dataset in alpacafarm bbh_shots codex gsm8k_shots tydiqa_shots mmlu_shots squad; do
    python scripts/combine_pickles.py --input_files rds_sels_qwen/**/${dataset}/*.pkl --output_file rds_sels_qwen/${dataset}_cossim.pkl
done

for size in 10000 25000 50000 100000 326000 700000 1200000 1500000 2500000; do
    python -m minimal_multitask.get_top_aggregated_influences \
        --input_files rds_sels_qwen/*_cossim.json \
        --output_file rds_sels_qwen/multitask_rrmax_${size}.json \
        --output_size $size --selection_method max \
        --train_dataset tulu_v2_unfiltered_data_dedup.jsonl \
        --output_dataset
done


# finetune on top 10k
shell_scripts/core_commands/model_training/weka/full_finetune.sh /weka/hamishi/influence-tuning/rds_sels_qwen/multitask_rrmax_10000.json qwen_sel_rrmax_10000
shell_scripts/core_commands/model_training/weka/full_finetune.sh /weka/hamishi/influence-tuning/rds_sels_qwen/multitask_rrmax_25000.json qwen_sel_rrmax_25000
shell_scripts/core_commands/model_training/weka/full_finetune.sh /weka/hamishi/influence-tuning/rds_sels_qwen/multitask_rrmax_50000.json qwen_sel_rrmax_50000
shell_scripts/core_commands/model_training/weka/full_finetune.sh /weka/hamishi/influence-tuning/rds_sels_qwen/multitask_rrmax_100000.json qwen_sel_rrmax_100000
shell_scripts/core_commands/model_training/weka/full_finetune_7b_multigpu.sh /weka/hamishi/influence-tuning/rds_sels_qwen/multitask_rrmax_326000.json qwen_sel_rrmax_326000
shell_scripts/core_commands/model_training/weka/full_finetune_7b_multigpu.sh /weka/hamishi/influence-tuning/rds_sels_qwen/multitask_rrmax_700000.json qwen_sel_rrmax_700000
shell_scripts/core_commands/model_training/weka/full_finetune_7b_multigpu.sh /weka/hamishi/influence-tuning/rds_sels_qwen/multitask_rrmax_1200000.json qwen_sel_rrmax_1200000
shell_scripts/core_commands/model_training/weka/full_finetune_7b_multigpu.sh /weka/hamishi/influence-tuning/rds_sels_qwen/multitask_rrmax_1500000.json qwen_sel_rrmax_1500000
shell_scripts/core_commands/model_training/weka/full_finetune_7b_multigpu.sh /weka/hamishi/influence-tuning/rds_sels_qwen/multitask_rrmax_2500000.json qwen_sel_rrmax_2500000