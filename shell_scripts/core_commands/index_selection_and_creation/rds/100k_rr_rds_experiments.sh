set -ex

for model in meta-llama/Llama-2-7b-hf allenai/OLMo-2-1124-7B allenai/OLMo-2-1124-13B meta-llama/Llama-2-13b-hf meta-llama/Llama-2-70b-hf meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-70B; do
    clean_model_name=$(echo $model | sed 's/\//-/g')
    folder=rds_100k_exp_from_base_weighted_mean_pool_${clean_model_name}
    mkdir -p $folder
    # construct train index with model
    python -m minimal_multitask.compute_influence_cosinesim \
        --model_name $model \
        --seed 42 \
        --train_dataset tulu_v2_unfiltered_data_dedup_balanced_100k.jsonl \
        --eval_dataset alpacafarm \
        --index_path ${folder}/cosine_train_reps.pt \
        --save_dir  ${folder}/ \
        --batch_size 1 \
        --pooling weighted_mean
    # construct eval index
    for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
        python -m minimal_multitask.compute_influence_cosinesim \
            --model_name allenai/OLMo-2-1124-7B \
            --seed 42 \
            --train_dataset tulu_v2_unfiltered_data_dedup_balanced_100k.jsonl \
            --eval_dataset $dataset \
            --index_path  ${folder}/cosine_train_reps.pt \
            --save_dir  ${folder}/ \
            --batch_size 1 \
            --pooling weighted_mean
    done
    # get top per-task influences
    for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
        python -m minimal_multitask.get_top_influences \
            --input_files  ${folder}/${dataset}_cossim.pkl \
            --output_file  ${folder}/${dataset}_top10k.json \
            --output_size 10000 --selection_method max \
            --train_datasets tulu_v2_unfiltered_data_dedup_balanced_100k.jsonl \
            --output_dataset
    done
    # get top rrmax influences
    # question is: does top rrmax beat random selection?
    # required for the setup to make sense.
    python -m minimal_multitask.get_top_aggregated_influences \
        --input_files  ${folder}/*_cossim.pkl \
        --output_file  ${folder}/multitask_rrmax_10k.json \
        --output_size 10000 --selection_method max \
        --train_dataset tulu_v2_unfiltered_data_dedup_balanced_100k.jsonl \
        --output_dataset
    exit
done
