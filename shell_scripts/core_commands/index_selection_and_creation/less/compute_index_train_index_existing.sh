# given we have made an index, select from it for all our evals.

for dataset in alpacafarm gsm8k_shots tydiqa_shots mmlu_shots codex bbh_shots squad; do
    # replace file with beaker ids 
    shards_dataset_file=$1
    while IFS=$'\t' read -r shard dataset_id; do
            echo "Processing $shard $dataset_id"
            gantry run \
                --workspace hamishivi \
                --cluster ai2/allennlp-cirrascale \
                --budget ai2/oe-adapt \
                --nfs \
                --allow-dirty --priority normal \
                --workspace ai2/minimal-multitask-finetuning \
                --gpus 1 \
                --env-secret HF_TOKEN=HF_TOKEN \
                --name less_warmup_hessian_select_${dataset}_${shard} \
                --task-name less_warmup_hessian_select_${dataset}_${shard} \
                --dataset "${dataset_id}:/index" \
                --env LD_LIBRARY_PATH=/opt/conda/envs/venv/lib \
                -- python -m minimal_multitask.compute_influence_train_index  \
                    --model_name /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/random-unfil-then-loras \
                    --top_k 1000000 \
                    --instance_to_influences /results/unfiltered_tulu_${dataset}_shots_samples_${shard}.pkl \
                    --seed 42 \
                    --random_transform 8192 \
                    --normalise_influences \
                    --eval_dataset ${dataset} \
                    --index_path /index/index_llama_2_128_lora_norm_tulu_unfiltered_unfiltered_tulu_shard_${shard}.faiss \
                    --llama_model \
                    --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered/subshards/unfiltered_tulu_shard_${shard}.jsonl \
                    --grad_batch 12
    done < "$shards_dataset_file"
done