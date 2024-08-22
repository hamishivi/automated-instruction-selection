BEAKER_PATH=$1
# SUBPATH=$3

# command for gantry here
# includes oai key and turning off alpaca eval 2 for alpaca eval stuff.
GANTRY_CMD="gantry run --cluster ai2/allennlp-cirrascale --budget ai2/oe-adapt --allow-dirty --priority normal --workspace ai2/minimal-multitask-finetuning --gpus 1 --env-secret OPENAI_API_KEY=OPENAI_API_KEY --env LD_LIBRARY_PATH=/opt/conda/envs/venv/lib --env IS_ALPACA_EVAL_2=False --dataset ${BEAKER_PATH}:/model"

for file in /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered/subshards/*.jsonl; do
    if [ -f "$file" ]; then
        shard=$(basename $file .jsonl)
            echo "Processing $shard"
            $GANTRY_CMD --name train_index_influence_less_${shard} --task-name train_index_influence_less_${shard} -- \
            python -m minimal_multitask.compute_influence_train_index \
                --model_name /model \
                --top_k 326154 \
                --seed 42 \
                --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered/subshards/${shard}.jsonl \
                --eval_dataset alpacafarm \
                --index_path /results/tulu_unfiltered_tulu_unfiltered_${shard}.faiss \
                --instance_to_influences /results/alpacafarm_tulu_unfiltered_tulu_unfiltered_${shard}.pkl \
                --save_index \
                --vanilla_gradients \
                --normalise_influences \
                --random_transform 8192 \
                --grad_batch 12 \
                --llama_model
    fi
done

# export LD_LIBRARY_PATH=/net/nfs.cirrascale/allennlp/hamishi/conda-envs/mmft/lib:$LD_LIBRARY_PATH

# python -m minimal_multitask.compute_influence_train_index \
#     --model_name ff_random_10k_rand_lora \
#     --tokenizer ff_random_10k \
#     --top_k 326154 \
#     --seed 42 \
#     --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered/tulu_v2_unfiltered_data_1k_sample.jsonl \
#     --eval_dataset alpacafarm \
#     --index_path tmp/index.faiss \
#     --instance_to_influences tmp/results.pkl \
#     --save_index \
#     --vanilla_gradients \
#     --normalise_influences \
#     --random_transform 8192 \
#     --grad_batch 12 \
#     --llama_model