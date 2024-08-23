# Influence Tuning: Instruction tuning guided by influence functions

## Quick start

Most of this respository assumes you are working on beaker internally Ai2. Generally, this just means the scripts will submit jobs to beaker automatically. If you want to run stuff locally, installing the dependencies in the `environment.yml` and `requirements.txt` should work, and then just running the commands in the scripts inside the gantry commands (i.e. everything after `-- python ...`).

Let me quickly go through an example workflow, which should touch most of the core scripts.

You will also need eval data and train data in a folder called `data`. Contact Hamish for this for now.

### Training

To train a model, you can use the scripts in `shell_scripts/core_commands/model_training`. The scripts named do the appropriate things. You have to provide a `.jsonl` file, which contains data formatted following the tulu format (i.e. like [the huggingface tulu 2 dataset](https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture)). You can also randomly subselect from this dataset, or train loras on top of a trained model, or train loras and full finetune. The scripts are mostly self-descriptive.

For example, to train llama 2 7b on a given file, I can run:
```bash
./shell_scripts/core_commands/model_training/full_finetune.sh <filename> <run name>
```
Or to train on x random samples from a file:
```bash
./shell_scripts/core_commands/model_training/random_select.sh <filename> <run name> <x>
```

You can also run locally with:
```bash
python -m minimal_multitask.instruction_tune \
        --model_name /model \
        --output_dir /results \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 128 \
        --num_train_epochs 2 \
        --learning_rate 2e-5 \
        --seed 42 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type linear \
        --weight_decay 0. \
        --evaluation_strategy no \
        --save_strategy no \
        --logging_steps 1 \
        --is_llama=True \
        --lora_alpha 512 \
        --lora_rank 128 \
        --use_hf_auth_token True \
        --train $TRAIN_FILE
```

Just replace the flags with what makes sense for you. If you remove the `lora_{alpha/rank}` flags then full-finetuning will happen.

### Index creation and selection

Given a trained model, we can then do index creation and selection, the examples of which live in `shell_scripts/core_commands/index_selection_and_creation`. There is support for both logix and less(+) style approaches here. Note for logix I'm using [my own fork](https://github.com/hamishivi/logix) due to having to make some fixes. I'll focus on Less-style since that's likely the core method we will use.

For less, we need a model trained with loras. Then we can run:
```bash
shell_scripts/core_commands/index_selection_and_creation/less/compute_index_train_index.sh <model_beaker_id>
```

`model_beaker_id` should point to the model in beaker. It should be easy to edit the script to point to models elsewhere. You can also run locally with:
```bash
export LD_LIBRARY_PATH=/net/nfs.cirrascale/allennlp/hamishi/conda-envs/mmft/lib:$LD_LIBRARY_PATH

python -m minimal_multitask.compute_influence_train_index \
    --model_name <model_name> \
    --tokenizer <tokenizer_name> \
    --top_k 326154 \
    --seed 42 \
    --train_dataset <train dataset> \
    --eval_dataset <eval dataset name to select over> \
    --index_path <index save path> \
    --instance_to_influences <influence score pickle to save as> \
    --save_index \
    --vanilla_gradients \
    --normalise_influences \
    --random_transform 8192 \
    --grad_batch 12 \
    --llama_model
```

Note that I usually use a sharded setup to compute the index, splitting up the dataset into ~29 files and individually computing indices. This is to parallelise the index computation.

To then run selection, we can either simply run `shell_scripts/core_commands/index_selection_and_creation/less/compute_index_train_index_existing.sh <path_to_shard_names.txt>`. `<path_to_shard_names.txt>` should be a path to a text file that has the format:
```
shard_name_1 beaker_id_1
shard_name_2 beaker_id_2
shard_name_3 beaker_id_3
shard_name_4 beaker_id_4
...
```

Alternatively, you can run:
```bash
export LD_LIBRARY_PATH=/net/nfs.cirrascale/allennlp/hamishi/conda-envs/mmft/lib:$LD_LIBRARY_PATH

python -m minimal_multitask.compute_influence_train_index \
    --model_name <model_name> \
    --tokenizer <tokenizer_name> \
    --top_k 326154 \
    --seed 42 \
    --train_dataset <train dataset> \
    --eval_dataset <eval dataset name to select over> \
    --index_path <path to index> \
    --instance_to_influences <influence score pickle to save as> \
    --vanilla_gradients \
    --normalise_influences \
    --random_transform 8192 \
    --grad_batch 12 \
    --llama_model
```

That is, you just run the same command as before, but with a different eval dataset! So long as you give the right path to the index you already made, this should be easy.

### Aggregation

Running the above gives us sets of `pickle` files that have the format `{ test_idx: {train_idx_1: influence_score_1, train_idx_2: influence_score_2, ...}, ...}`. We then need to select data from these. In addition, we also need to aggregate scores from multiple pickle files if we have run multiple shards (often with very large datasets). To do this, we can run:

```bash
# get the datasets from all experiments with `beaker_prefix`, e.g. all jobs outputting the given sharded pickle
python minimal_multitask/get_shard_file.py --prefix <beaker_prefix> --outfile <output_path>
# download the pickles using the list from before, and then do the selection
./shell_scripts/core_commands/aggregate/download_and_select_from_pickles.sh <pickle_save_location> <previous_output_path>
```

This will put a file called `top_10k.json` at `<previous_output_path>`, and then you can train on it! You can also use the pickle files for analysing files.

### Analysis

Generally, analysis scripts live in `scripts`. These are undocumented unless Hamish decides to write more about them. I also have a tonne of old scripts from a previous version of this project in `scripts/old`.
