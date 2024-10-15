set -ex

python -m minimal_multitask.compute_influence_sentence_embedding \
    --seed 42 \
    --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/200k_exps/unbalanced_distribution_200k.jsonl \
    --eval_dataset alpacafarm \
    --index_path sentence_embedding_200k_exp/cosine_train_reps.pt \
    --save_dir sentence_embedding_200k_exp/ \
    --batch_size 1

for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
    python -m minimal_multitask.compute_influence_sentence_embedding \
        --seed 42 \
        --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/200k_exps/unbalanced_distribution_200k.jsonl \
        --eval_dataset $dataset \
        --index_path sentence_embedding_200k_exp/cosine_train_reps.pt \
        --save_dir sentence_embedding_200k_exp/ \
        --batch_size 1
done

for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
    python -m minimal_multitask.get_top_influences \
        --input_files sentence_embedding_200k_exp/${dataset}_embedding.pkl \
        --output_file sentence_embedding_200k_exp/${dataset}_top10k.json \
        --output_size 10000 --selection_method max \
        --train_datasets /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/200k_exps/unbalanced_distribution_200k.jsonl \
        --output_dataset
done