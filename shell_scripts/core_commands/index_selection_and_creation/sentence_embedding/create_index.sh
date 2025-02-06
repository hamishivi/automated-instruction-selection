set -ex

# python -m minimal_multitask.compute_influence_sentence_embedding \
#     --seed 42 \
#     --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/200k_exps/unbalanced_distribution_200k.jsonl \
#     --eval_dataset alpacafarm \
#     --index_path sentence_embedding_200k_exp/cosine_train_reps.pt \
#     --save_dir sentence_embedding_200k_exp/ \
#     --batch_size 1

for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
    python -m minimal_multitask.compute_influence_sentence_embedding \
        --seed 42 \
        --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/200k_exps/unbalanced_distribution_200k.jsonl \
        --eval_dataset $dataset \
        --index_path ccds_embedding_200k_exp/t5base_cosine_train_reps.pt \
        --save_dir ccds_embedding_200k_exp/ \
        --batch_size 16 \
        --model_name_or_path sentence-transformers/gtr-t5-base
done

for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
    python -m minimal_multitask.get_top_influences \
        --input_files ccds_embedding_200k_exp/${dataset}_embedding.pkl \
        --output_file selections/ccds_embedding_200k_exp/t5base_${dataset}_meanmax10k.json \
        --output_size 10000 --selection_method mean_max \
        --train_datasets /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/200k_exps/unbalanced_distribution_200k.jsonl \
        --output_dataset
done