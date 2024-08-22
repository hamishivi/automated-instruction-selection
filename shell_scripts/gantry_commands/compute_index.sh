# add conda to ld path
export LD_LIBRARY_PATH=/net/nfs.cirrascale/allennlp/hamishi/conda-envs/mmft/lib:$LD_LIBRARY_PATH

# start with alpacafarm, create index
mkdir results/llama_2_128lora_alpaca

python -m minimal_multitask.compute_influence_train_index \
    --model_name results/llama_2_128lora_tulu2_dataset \
    --top_k 326154 \
    --instance_to_influences results/llama_2_128lora_tulu2/alpacafarm.pkl \
    --seed 42 \
    --random_transform 8192 \
    --normalise_influences \
    --eval_dataset alpacafarm \
    --index_path index_llama_2_128_lora_norm_tulu2.faiss \
    --vanilla_gradients \
    --save_index \
    --llama_model \
    --train_dataset tulu2 \
    --grad_batch 12

# now, over all eval sets. Don't save index, just reload

python -m minimal_multitask.compute_influence_train_index \
    --model_name results/llama_2_128lora_tulu2_dataset \
    --top_k 49881 \
    --instance_to_influences results/llama_2_128_lora_tydiqa_train/tydiqa_train_set.pkl \
    --seed 42 \
    --random_transform 8192 \
    --normalise_influences \
    --eval_dataset tydiqa_shots \
    --index_path indices/index_llama_2_128_lora_norm_tydiqa_train.faiss \
    --vanilla_gradients \
    --save_index \
    --llama_model \
    --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tydiqa/tydiqa_data.jsonl \
    --grad_batch 12

python -m minimal_multitask.compute_influence_train_index \
    --model_name results/llama_2_128lora_tulu2_dataset \
    --top_k 7473 \
    --instance_to_influences results/llama_2_128_lora_gsm8k_train/gsm8k_train_set.pkl \
    --seed 42 \
    --random_transform 8192 \
    --normalise_influences \
    --eval_dataset gsm8k_shots \
    --index_path indices/index_llama_2_128_lora_norm_gsm8k_train.faiss \
    --vanilla_gradients \
    --save_index \
    --llama_model \
    --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/gsm8k/gsm8k_data.jsonl \
    --grad_batch 12


python -m minimal_multitask.compute_influence_train_index \
    --model_name results/llama_2_128lora_tulu2_dataset \
    --top_k 87599 \
    --instance_to_influences results/llama_2_128_lora_squad_train/squad_train_set.pkl \
    --seed 42 \
    --random_transform 8192 \
    --normalise_influences \
    --eval_dataset squad \
    --index_path indices/index_llama_2_128_lora_norm_squad_train.faiss \
    --vanilla_gradients \
    --save_index \
    --llama_model \
    --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/squad/squad_data.jsonl \
    --grad_batch 12



for i in {1..6}
do
    gantry run --workspace hamishivi --cluster ai2/allennlp-cirrascale --budget ai2/oe-adapt --allow-dirty --priority normal --workspace ai2/minimal-multitask-finetuning --gpus 1 --env-secret HF_TOKEN=HF_TOKEN --name tulu_v2_unfiltered_data_300k_construct_index_shard${i} --task-name tulu_v2_unfiltered_data_300k_construct_index_shard${i} --env LD_LIBRARY_PATH=/opt/conda/envs/venv/lib \
        -- python -m minimal_multitask.compute_influence_train_index  \
        --model_name /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_v2_unfiltered_data_300k_sample_lora \
        --top_k 1000000 \
        --instance_to_influences /results/unfiltered_tulu_alpacafarm_samples_shard${i}.pkl \
        --seed 42 \
        --random_transform 8192 \
        --normalise_influences \
        --eval_dataset alpacafarm \
        --index_path /results/index_llama_2_128_lora_norm_tulu_unfiltered_shard${i}.faiss \
        --vanilla_gradients \
        --save_index \
        --llama_model \
        --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered/tulu_v2_unfiltered_data_shard${i}.jsonl \
        --grad_batch 12 \
        --add_pad_before_load meta-llama/Llama-2-7b-hf \
        --tokenizer meta-llama/Llama-2-7b-hf
done

gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots

i=1_5
for eval_dataset in codex mmlu_shots
do
    gantry run \
        --workspace hamishivi \
        --cluster ai2/allennlp-cirrascale \
        --budget ai2/oe-adapt \
        --allow-dirty --priority normal \
        --workspace ai2/minimal-multitask-finetuning \
        --gpus 1 --env-secret HF_TOKEN=HF_TOKEN \
        --name select_${eval_dataset}_${i} \
        --task-name select_${eval_dataset}_${i} \
        --env LD_LIBRARY_PATH=/opt/conda/envs/venv/lib \
        --dataset '01HW4KD1SDEBVME0EKBQ55V04T:/index' \
        -- python -m minimal_multitask.compute_influence_train_index  \
            --model_name /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/results/llama_2_128lora_tulu2_dataset \
            --top_k 1000000 \
            --instance_to_influences /results/unfiltered_tulu_${eval_dataset}_samples_shard${i}.pkl \
            --seed 42 \
            --random_transform 8192 \
            --normalise_influences \
            --eval_dataset ${eval_dataset} \
            --index_path /index/index_llama_2_128_lora_norm_tulu_unfiltered_shard${i}.faiss \
            --vanilla_gradients \
            --llama_model \
            --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered/tulu_v2_unfiltered_data_shard${i}.jsonl \
            --grad_batch 12 
done

for file in /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered/subshards/*.jsonl; do
    if [ -f "$file" ]; then
        shard=$(basename $file .jsonl)
        echo "Processing $shard"
        gantry run \
            --workspace hamishivi \
            --cluster ai2/allennlp-cirrascale \
            --budget ai2/oe-adapt \
            --allow-dirty --priority normal \
            --workspace ai2/minimal-multitask-finetuning \
            --gpus 1 --env-secret HF_TOKEN=HF_TOKEN \
            --name construct_index_from_random_unfil_train_${shard} \
            --task-name construct_index_from_random_unfil_train_${shard} \
            --env LD_LIBRARY_PATH=/opt/conda/envs/venv/lib \
            -- python -m minimal_multitask.compute_influence_train_index  \
                --model_name /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/random-unfil-then-loras \
                --top_k 1000000 \
                --instance_to_influences /results/unfiltered_tulu_alpacafarm_samples_${shard}.pkl \
                --seed 42 \
                --random_transform 8192 \
                --normalise_influences \
                --eval_dataset alpacafarm \
                --index_path /results/index_llama_2_128_lora_norm_tulu_unfiltered_${shard}.faiss \
                --vanilla_gradients \
                --save_index \
                --llama_model \
                --train_dataset $file \
                --grad_batch 12
    fi
done

# just select for gsm8k
for dataset in alpacafarm gsm8k_shots tydiqa_shots mmlu_shots codex bbh_shots squad; do
    shards_dataset_file="/net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/random_then_alpacaeval_select/beaker_ids_random_second_round.txt"
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