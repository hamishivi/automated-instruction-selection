set -ex

mkdir -p rds_200k_exp_from_base_alpaca
python -m minimal_multitask.compute_influence_cosinesim \
    --model_name meta-llama/Llama-2-7b-hf \
    --seed 42 \
    --train_dataset data/stanford_alpaca_data.jsonl \
    --eval_dataset alpacafarm \
    --index_path rds_200k_exp_from_base_alpaca/cosine_train_reps.pt \
    --save_dir rds_200k_exp_from_base_alpaca/ \
    --batch_size 1

for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
    python -m minimal_multitask.compute_influence_cosinesim \
        --model_name meta-llama/Llama-2-7b-hf \
        --seed 42 \
        --train_dataset data/stanford_alpaca_data.jsonl \
        --eval_dataset $dataset \
        --index_path rds_200k_exp_from_base_alpaca/cosine_train_reps.pt \
        --save_dir rds_200k_exp_from_base_alpaca/ \
        --batch_size 1
done


mkdir -p selection_files/rds_200k_exp_from_base_alpaca
for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
    python -m minimal_multitask.get_top_influences \
        --input_files rds_200k_exp_from_base_alpaca/${dataset}_cossim.pkl \
        --output_file selection_files/rds_200k_exp_from_base_alpaca/${dataset}_top10k.json \
        --output_size 10000 --selection_method max \
        --train_datasets data/stanford_alpaca_data.jsonl \
        --output_dataset
    shell_scripts/core_commands/model_training/full_finetune.sh /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/selection_files/rds_200k_exp_from_base_alpaca/${dataset}_top10k.json alpaca_${dataset}_test
done