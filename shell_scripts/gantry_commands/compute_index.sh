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

