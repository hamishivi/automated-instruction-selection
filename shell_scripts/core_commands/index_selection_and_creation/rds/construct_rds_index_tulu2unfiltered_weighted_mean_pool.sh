set -ex

mkdir -p rds_tulu2unfiltered_exp_from_base_weighted_mean_pool

for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
    python -m minimal_multitask.compute_influence_cosinesim \
        --model_name meta-llama/Llama-2-7b-hf \
        --seed 42 \
        --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/tulu_v2_unfiltered_data_dedup.jsonl \
        --eval_dataset $dataset \
        --index_path rds_tulu2unfiltered_exp_from_base_weighted_mean_pool/cosine_train_reps.pt \
        --save_dir rds_tulu2unfiltered_exp_from_base_weighted_mean_pool/ \
        --batch_size 1 \
        --pooling weighted_mean
done


for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
    python -m minimal_multitask.get_top_influences \
        --input_files rds_pickles/${dataset}.json \
        --output_file selections/rds_tulu2_unfiltered/${dataset}_top326k.json \
        --output_size 326000 --selection_method max \
        --train_datasets /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/tulu_v2_unfiltered_data_dedup.jsonl \
        --output_dataset
done