export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/


python -m minimal_multitask.compute_influence_train_index \
    --model_name results/alpaca_7b/full_finetune \
    --top_k 52002 \
    --instance_to_influences results/alpaca_7b/full_finetune_random_${i}.pkl \
    --seed 1 \
    --eval_dataset mmlu \
    --random_transform 8192 \
    --save_index \
    --index_path 8192_random_full_finetune_index.pkl 
