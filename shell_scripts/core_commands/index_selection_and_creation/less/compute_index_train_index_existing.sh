# given we have made an index, select from it for all our evals.
model_path=01J6GMQH7BKQRQ6FX7ZTJ0JWFM

for dataset in alpacafarm squad mmlu_shots codex bbh_shots tydiqa_shots gsm8k_shots; do
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
                --name lora_ff_10k_fixed_sampling_${dataset}_${shard} \
                --task-name lora_ff_10k_fixed_sampling_${dataset}_${shard} \
                --dataset "${dataset_id}:/index" \
                --dataset "${model_path}:/model" \
                --dataset 01J6QSXVDS4MN0W45HB2MHWXQN:/data \
                --env LD_LIBRARY_PATH=/opt/conda/envs/venv/lib \
                -- python -m minimal_multitask.compute_influence_train_index  \
                    --model_name /model \
                    --underlying_model_name /model/underlying_model \
                    --top_k 1000000 \
                    --instance_to_influences /results/unfiltered_tulu_${dataset}_shots_samples_${shard}.pkl \
                    --seed 42 \
                    --random_transform 8192 \
                    --normalise_influences \
                    --vanilla_gradients \
                    --eval_dataset ${dataset} \
                    --index_path /index/tulu_unfiltered_tulu_unfiltered_${shard}.faiss \
                    --llama_model \
                    --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/subshards/${shard}.jsonl \
                    --grad_batch 12
    done < "$shards_dataset_file"
done