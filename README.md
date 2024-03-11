# Influence Tuning

Very quick rundown: there is a lot of stuff in here from an older project, don't worry about it.

Look at the scripts in `shell_scripts/gantry` for how to run an influence experiment. These are configured for AI2 compute, but the basic commands are the same. Note you can use either Tulu or Alpaca as the train dataset.

Let me go through an example:

First, We set a seed value with `export i=1` or similar.

Then, we can instruction tune with loras, and save the result.
```bash
--priority normal --env-secret HF_TOKEN=HF_TOKEN --gpus 1 --name $EXP_NAME --task-name $EXP_NAME -- python -m minimal_multitask.instruction_tune \
    --model_name meta-llama/Llama-2-7b-hf \
    --output_dir model_${i} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --seed 1 \
    --use_hf_auth_token True \
    --save_strategy no \
    --evaluation_strategy no \
    --logging_steps 1 \
    --is_llama=True \
    --train tulu2 \
    --lora_rank 128 \
    --lora_alpha 512
```

Then, we can compute the influence. This method calculates all the train gradients and puts them into a FAISS index for reference, which can be saved to disk or just kept in memory. There is also `compute_influence_query_batching`, which instead computes the test gradients + hessian products, and then one by one calculates the train gradient and computes the influence, before saving. The latter is better if you are disk-space limited, but takes longer to re-run (since there is no saving of results).
```bash
python -m minimal_multitask.compute_influence_train_index \
    --model_name model_${i} \
    --top_k <dataset> \
    --instance_to_influences ${eval_dataset}.pkl \
    --seed 42 \
    --random_transform 8192 \
    --normalise_influences \
    --eval_dataset ${eval_dataset} \
    --index_path index.faiss \
    --vanilla_gradients \
    --save_index \
    --llama_model \
    --train_dataset tulu2 \
    --grad_batch 12
```
The list of evaluation datasets to use is in `minimal_multitask/data.py` (`DATASETS`). For the numbers reported, we use the `shots` variant of the dataset unless there is not one.


Then, given a pickle with some saved influence scores, we extract the top-k. This script has some flags for adjusting what algorithm to use, but we normally use top-k.
```bash
python -m minimal_multitask.get_top_influences \
        --input_file ${eval_dataset}.pkl \
        --output_file top_10k_${eval_dataset}_top_min.json \
        --output_size 10000 \
        --selection_method min \
        --train_dataset tulu2
```

We then fully-finetune on the chosen data.
```bash
python -m minimal_multitask.instruction_tune \
        --model_name meta-llama/Llama-2-7b-hf \
        --output_dir model_${eval_dataset}_${i} \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 128 \
        --num_train_epochs 2 \
        --learning_rate 2e-5 \
        --seed 42 \
        --evaluation_strategy no \
        --save_strategy no \
        --logging_steps 1 \
        --is_llama=True \
        --use_hf_auth_token True \
        --train tulu2 \
        --saved_instances top_10k_${eval_dataset}_top_min.json
```

Finally, we evaluate on the task. See `shell_scripts/gantry_commands/eval.sh` for the evaluation commands we use. You can get the evaluation data from `gs://hamishi-dev/mmft/eval_data`. Place it in `data/eval`, and modify the given paths accordingly. Any missing data can be gotten by running the evaluation data download script from the Open-Instruct Repo.

To train a model on random data, we can use the `--random_select k` flag on the `instruction_tune` script, where k is the number of random samples to pick.

## Aggregated Results

To aggregate results, given a folder with `.pkl` files generated using the above steps, run:
```bash
python -m minimal_multitask.get_top_aggregated_influences \
    --input_files folder/*.pkl \
    --output_file aggregated.json \
    --selection_method min \
    --output_size 10000 \
    --train_dataset tulu \
    --aggregation_method minmax
```
Use `--aggregation_method mean` for the top-mean results.

## Cosine Embeddings