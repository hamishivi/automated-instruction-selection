FOLDERS=(
    rds_100k_exp_from_base_weighted_mean_pool_meta-llama-Llama-2-7b-hf
    rds_100k_exp_from_base_weighted_mean_pool_meta-llama-Llama-2-13b-hf
    rds_100k_exp_from_base_weighted_mean_pool_meta-llama-Llama-2-70b-hf
    rds_100k_exp_from_base_weighted_mean_pool_meta-llama-Llama-3.1-8B
    rds_100k_exp_from_base_weighted_mean_pool_meta-llama-Llama-3.1-70B
    rds_100k_exp_from_base_weighted_mean_pool_allenai-OLMo-2-1124-7B
    rds_100k_exp_from_base_weighted_mean_pool_allenai-OLMo-2-1124-13B
)

for folder in ${FOLDERS[@]}; do
    filepath=/weka/hamishi/influence-tuning/${folder}/multitask_rrmax_10k.json
    ./shell_scripts/core_commands/model_training/weka/full_finetune.sh $filepath training_7b_${folder}
    ./shell_scripts/core_commands/model_training/weka/full_finetune_13b.sh $filepath training_13b_${folder}
    ./shell_scripts/core_commands/model_training/weka/full_finetune_llama_3_8b.sh $filepath training_llama_3_8b_${folder}
    ./shell_scripts/core_commands/model_training/weka/full_finetune_olmo_2_7b.sh $filepath training_olmo_2_7b_${folder}
    ./shell_scripts/core_commands/model_training/weka/full_finetune_olmo_2_13b.sh $filepath training_olmo_2_13b_${folder}
done
