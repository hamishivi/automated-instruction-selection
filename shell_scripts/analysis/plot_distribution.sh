# python3 scripts/compute_source_bars.py --normalize_count --output_file visualizations/selection_distribution/$1.png \
#     --input_files ../../hamishi/minimal-multitask-tuning/selection_files/lora_ff_10k_fixed_sampling/*/top_10k.json \
#     selections/lora_ff_10k_fixed_sampling/multitask_meanmin10k.json selections/lora_ff_10k_fixed_sampling/multitask_meanmean10k.json \
#     selections/tulu2_unfiltered_subsample/random_67k.json \
#     selections/lora_ff_10k_fixed_sampling/wildchat_normalizedmean10k_withsample.json \
#     selections/lora_ff_10k_fixed_sampling/arenahard_normalizedmean10k_withsample.json 

# Selection of "alpacaeval", "gsm8k", "tydiqa", "bbh", "mmlu", "codex", "squad"
python3 scripts/pairwise_source_bar.py --normalize_count --output_file visualizations/selection_distribution/grad_vs_rds.png \
    --input_files ../../hamishi/minimal-multitask-tuning/selection_files/lora_ff_10k_fixed_sampling/*/top_10k.json \
    --second_input_files selections/rds_tulu2_unfiltered/*_top10k.json 