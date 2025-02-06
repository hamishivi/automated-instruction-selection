set -ex

mkdir -p rds_pickles

for file in /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/subshards/*.jsonl; do
    if [ -f "$file" ]; then
        shard=$(basename $file .jsonl)
        echo "Processing $shard"
        shardnum="${shard##*_}"
        ehco "Processing shard num $shardnum"
        gantry run \
            --workspace hamishivi \
            --cluster ai2/allennlp-cirrascale \
            --cluster ai2/pluto-cirrascale \
            --budget ai2/oe-adapt \
            --allow-dirty \
            --priority normal \
            --workspace ai2/minimal-multitask-finetuning \
            --gpus 1 \
            --env-secret HF_TOKEN=$HF_TOKEN \
            --name rds_tulu2unfiltered_exp_from_base_weighted_mean_pool_${shardnum} \
            --task-name rds_tulu2unfiltered_exp_from_base_weighted_mean_pool_${shardnum} \
            -- 
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
done

for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
    python -m scripts.combine_pickles --input_files rds_pickles/rds_tulu2unfiltered_exp_from_base_weighted_mean_pool_*/${dataset}_cossim.pkl \
                                        --output_file rds_pickles/${dataset}_cossim.pkl
done

for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
    python -m minimal_multitask.get_top_influences \
        --input_files rds_pickles/${dataset}_cossim.json \
        --output_file selections/rds_tulu2_unfiltered/${dataset}_top10k.json \
        --output_size 10000 --selection_method max \
        --train_datasets /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/tulu_v2_unfiltered_data_dedup.jsonl \
        --output_dataset
done