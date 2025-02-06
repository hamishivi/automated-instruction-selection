
set -ex

shard=$1
shardnum="${shard##*_}"
EXP_NAME=rds_tulu2unfiltered_exp_from_base_weighted_mean_pool_${shardnum}

echo "Processing shard num $shardnum"
gantry run \
        --workspace hamishivi \
        --cluster ai2/allennlp-cirrascale \
        --cluster ai2/pluto-cirrascale \
        --budget ai2/oe-adapt \
        --allow-dirty \
        --priority normal \
        --workspace ai2/minimal-multitask-finetuning \
        --gpus 1 \
        --env-secret HF_TOKEN=HF_TOKEN \
        --name $EXP_NAME \
        --task-name $EXP_NAME \
        -- bash -c "for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
                python -m minimal_multitask.compute_influence_cosinesim \
                    --model_name meta-llama/Llama-2-7b-hf \
                    --seed 42 \
                    --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/subshards/${shard}.jsonl \
                    --eval_dataset $dataset \
                    --index_path rds_pickles/rds_tulu2unfiltered_exp_from_base_weighted_mean_pool_${shardnum}/cosine_train_reps.pt \
                    --save_dir rds_pickles/rds_tulu2unfiltered_exp_from_base_weighted_mean_pool_${shardnum}/ \
                    --batch_size 1 \
                    --pooling weighted_mean
            done"

# beaker session create --gpus 1 --budget ai2/oe-adapt
# bash shell_scripts/core_commands/index_selection_and_creation/rds/rds_tulu2sharded_beaker.sh tulu_v2_unfiltered_data_dedup_aa