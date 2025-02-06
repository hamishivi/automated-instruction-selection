set -ex

mkdir -p rds_200k_exp_from_base_label_only
python -m minimal_multitask.compute_influence_cosinesim \
    --model_name meta-llama/Llama-2-7b-hf \
    --seed 42 \
    --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/200k_exps/unbalanced_distribution_200k.jsonl \
    --eval_dataset alpacafarm \
    --index_path rds_200k_exp_from_base_label_only/cosine_train_reps.pt \
    --save_dir rds_200k_exp_from_base_label_only/ \
    --batch_size 1 \
    --label_only

for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
    python -m minimal_multitask.compute_influence_cosinesim \
        --model_name meta-llama/Llama-2-7b-hf \
        --seed 42 \
        --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/200k_exps/unbalanced_distribution_200k.jsonl \
        --eval_dataset $dataset \
        --index_path rds_200k_exp_from_base_label_only/cosine_train_reps.pt \
        --save_dir rds_200k_exp_from_base_label_only/ \
        --batch_size 1 \
        --label_only
done


mkdir -p selection_files/rds_200k_exp_from_base
for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
    python -m minimal_multitask.get_top_influences \
        --input_files rds_200k_exp_from_base_label_only/${dataset}_cossim_labelonly.pkl \
        --output_file rds_200k_exp_from_base_label_only/${dataset}_top10k.json \
        --output_size 10000 --selection_method max \
        --train_datasets /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/200k_exps/unbalanced_distribution_200k.jsonl \
        --output_dataset
done