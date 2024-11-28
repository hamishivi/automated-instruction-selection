set -ex

mkdir -p rds_100k_exp_from_base_weighted_mean_pool_olmo2_13b

# mkdir -p rds_200k_exp_from_base
python -m minimal_multitask.compute_influence_cosinesim \
    --model_name allenai/OLMo-2-1124-13b \
    --seed 42 \
    --train_dataset tulu_v2_unfiltered_data_dedup_balanced_100k.jsonl \
    --eval_dataset alpacafarm \
    --index_path rds_100k_exp_from_base_weighted_mean_pool_olmo2_13b/cosine_train_reps.pt \
    --save_dir rds_100k_exp_from_base_weighted_mean_pool_olmo2_13b/ \
    --batch_size 1 \
    --pooling weighted_mean

for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
    python -m minimal_multitask.compute_influence_cosinesim \
        --model_name allenai/OLMo-2-1124-13b \
        --seed 42 \
        --train_dataset tulu_v2_unfiltered_data_dedup_balanced_100k.jsonl \
        --eval_dataset $dataset \
        --index_path rds_100k_exp_from_base_weighted_mean_pool_olmo2_13b/cosine_train_reps.pt \
        --save_dir rds_100k_exp_from_base_weighted_mean_pool_olmo2_13b/ \
        --batch_size 1 \
        --pooling weighted_mean
done


for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
    python -m minimal_multitask.get_top_influences \
        --input_files rds_100k_exp_from_base_weighted_mean_pool_olmo2_13b/${dataset}_cossim.pkl \
        --output_file rds_100k_exp_from_base_weighted_mean_pool_olmo2_13b/${dataset}_top10k.json \
        --output_size 10000 --selection_method max \
        --train_datasets tulu_v2_unfiltered_data_dedup_balanced_100k.jsonl \
        --output_dataset
done