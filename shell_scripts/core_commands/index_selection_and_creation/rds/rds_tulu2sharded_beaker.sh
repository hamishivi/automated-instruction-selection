
set -ex

shard=$1
shardnum="${shard##*_}"
echo "Processing shard num $shardnum"
for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
    python -m minimal_multitask.compute_influence_cosinesim \
        --model_name meta-llama/Llama-2-7b-hf \
        --seed 42 \
        --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/subshards/${shard}.jsonl \
        --eval_dataset $dataset \
        --index_path rds_pickles/rds_tulu2unfiltered_exp_from_base_weighted_mean_pool_${shardnum}/cosine_train_reps.pt \
        --save_dir rds_pickles/rds_tulu2unfiltered_exp_from_base_weighted_mean_pool_${shardnum}/ \
        --batch_size 1 \
        --pooling weighted_mean
done

# beaker session create --gpus 1 --budget ai2/oe-adapt
# bash shell_scripts/core_commands/index_selection_and_creation/rds/rds_tulu2sharded_beaker.sh tulu_v2_unfiltered_data_dedup_ba

python -m minimal_multitask.compute_influence_cosinesim \
    --model_name meta-llama/Llama-2-7b-hf \
    --seed 42 \
    --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/subshards/tulu_v2_unfiltered_data_dedup_${shardnum}.jsonl \
    --eval_dataset data/customized_datasets/tulu2_1k.jsonl \
    --index_path rds_pickles/rds_tulu2unfiltered_exp_from_base_weighted_mean_pool_${shardnum}/cosine_train_reps.pt \
    --save_dir rds_pickles/rds_tulu2unfiltered_exp_from_base_weighted_mean_pool_${shardnum}/ \
    --batch_size 1 \
    --pooling weighted_mean

python -m minimal_multitask.eval.bbh.run_eval \
    --data_dir /data/eval/bbh \
    --save_dir temp \
    --model_name_or_path 'meta-llama/Llama-2-7b-hf' \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format