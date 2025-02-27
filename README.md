# Automated Instruction Selection

This is the repository associated with the paper [Practical Large-Scale Data Selection for Instruction Tuning](https://todo).

![performance of RDS+ compared to other methods and when selecting datasets of increasing size](performance_graphic.png)

We release data and models associated with this paper on HuggingFace at [this collection](https://huggingface.co/collections/hamishivi/practical-large-scale-data-selection-for-instruction-tuning-677d7e8ca0295426c1915930).

## Install Dependencies

Note I assume you are working in a Linux environment with CUDA-compatible GPUs and conda.

Install the requirements in `environment.yml` and `requirements.txt`:
```bash
conda env create --file environment.yml --name mmft
conda activate mmft
pip install -r requirements.txt
```

Download the required data files (for eval and training):
```bash
./shell_scripts/download_eval_data.sh
./shell_scripts/download_train_data.sh  # optional if you don't want to do data selection on Tulu datasets.
```

Now you should be good to go!
If you are at Ai2, then you don't even need these steps, since we have a bunch of [beaker-gantry](https://github.com/allenai/beaker-gantry) scripts that should automatically setup the environment when you run them.

## Run the Selection and Training Pipeline

The pipeline for this paper has four steps, described below:
1. Construct an index for training and eval data.
2. Select data based off the score.
3. Train a model on the selected data.
4. Evaluate the trained model.

Below, we detail the steps when selecting and training data for the **RDS+** method. See "Other selection methods" below for how we ran baselines.

### Index creation and selection

Scripts for doing indexing and selection live in `shell_scripts/core_commands/index_selection_and_creation`. I have support for a few different approaches here, but the most important one is RDS+ (see our paper for more details on this).

We start by indexing our training data, using a given base model:
```bash
python -m minimal_multitask.compute_influence_cosinesim \
    --model_name meta-llama/Llama-2-7b-hf \
    --seed 42 \
    --train_dataset data/training_data/tulu_v2_unfiltered_data_dedup.jsonl \
    --eval_dataset alpacafarm \
    --index_path rds_200k_exp_from_base_weighted_mean_pool/cosine_train_reps.pt \
    --save_dir rds_200k_exp_from_base_weighted_mean_pool/ \
    --batch_size 1 \
    --pooling weighted_mean
```
This gives us an index we can re-use for future queries. We then query for the eval set we care about, loading this train index:
```bash
for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
    python -m minimal_multitask.compute_influence_cosinesim \
        --model_name meta-llama/Llama-2-7b-hf \
        --seed 42 \
        --train_dataset data/training_data/tulu_v2_unfiltered_data_dedup.jsonl \
        --eval_dataset $dataset \
        --index_path rds_200k_exp_from_base_weighted_mean_pool/cosine_train_reps.pt \
        --save_dir rds_200k_exp_from_base_weighted_mean_pool_${dataset}/ \
        --batch_size 1 \
        --pooling weighted_mean
done
```
Finally, we want to use the computed scores to produce selected datasets for us:
```bash
for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
    python -m minimal_multitask.get_top_influences \
        --input_files rds_200k_exp_from_base_weighted_mean_pool/${dataset}_cossim.pkl \
        --output_file rds_200k_exp_from_base_weighted_mean_pool/${dataset}_top10k.json \
        --output_size 10000 --selection_method max \
        --train_datasets training_data/tulu_v2_unfiltered_data_dedup.jsonl \
        --output_dataset
done
```
We can then directly finetune on the resulting dataset with e.g.:
```bash
./shell_scripts/core_commands/model_training/full_finetune.sh rds_200k_exp_from_base_weighted_mean_pool/alpacafarm_top10k.json alpacafarm_exp
```

#### Sharded Indexing

Some of the datasets we indexed are quite large, and so to process these efficiently we relied on sharding them. We provide easy ways to do this in this codebase.

First, shard your dataset (e.g. using the `split` command): `split -l 200000 <filename>`.
Pick your shard size based on how many gpus you can run in parallel.

Then, for each shard, run the training data and eval data indexing:
```bash
for file in shards/*.jsonl; do
    shard=$(basename $file .jsonl)
    echo "Processing $shard"
    python -m minimal_multitask.compute_influence_cosinesim \
        --model_name meta-llama/Llama-2-7b-hf \
        --seed 42 \
        --train_dataset $file \
        --eval_dataset alpacafarm \
        --index_path rds_selection_${shard}/cosine_train_reps.pt \
        --save_dir rds_selection_${shard}/ \
        --batch_size 1 \
        --pooling weighted_mean
done

for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
    for file in shards/*.jsonl; do
        shard=$(basename $file .jsonl)
        echo "Processing $shard"
        python -m minimal_multitask.compute_influence_cosinesim \
            --model_name meta-llama/Llama-2-7b-hf \
            --seed 42 \
            --train_dataset $file \
            --eval_dataset $dataset \
            --index_path rds_selection_${shard}/cosine_train_reps.pt \
            --save_dir rds_selection/${dataset}/${shard}/ \
            --batch_size 1 \
            --pooling weighted_mean
    done
done
```

We then combine the sharded scores into one file. Note this takes a lot of RAM:
```bash
for dataset in alpacafarm bbh_shots codex gsm8k_shots tydiqa_shots mmlu_shots squad arena_hard wildchat; do
    python scripts/combine_pickles.py --input_files rds_selection/${dataset}/**/*.pkl --output_file rds_selection/${dataset}_cossim.pkl
done
```

Finally, given this we can then select datasets. I also have some more optimized selection code that helps when selecting large datasets.
Note you need the full datafile around to construct the train file.
```bash
for dataset in alpacafarm bbh_shots codex gsm8k_shots tydiqa_shots mmlu_shots squad arena_hard wildchat; do
    python -m minimal_multitask.get_top_optimized \
        --input_files rds_selection/${dataset}_cossim.json \
        --output_file rds_selection/${dataset}_top10k.json \
        --output_size 10000 --selection_method max \
        --train_datasets <original full datafile>
done

# or if you want to use the round-robin setup
python -m minimal_multitask.get_top_aggregated_influences \
    --input_files rds_selection/*_cossim.json \
    --output_file rds_selection/multitask_rrmax_320k.json \
    --output_size 320000 --selection_method max \
    --train_dataset <original full datafile> \
    --output_dataset
```

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
Please look into the scripts and feel free to edit them (in particular, you might wish to edit the output directory name).

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

If you are at Ai2, you can use gantry with these scripts by setting `GANTRY=1` before running the script, e.g:
```bash
GANTRY=1 ./shell_scripts/core_commands/model_training/full_finetune.sh <filename> <run name>
```


### Evaluation

Given a trained model, you can evaluate using `eval/eval.sh <model_path>`. Look into that script if you want to run a particular eval.
We base our evaluations off the implementations in [Open-Instruct](https://github.com/allenai/open-instruct).
**Note: for non-llama 2 models, please replace `create_prompt_with_tulu_chat_format` with `create_prompt_with_huggingface_tokenizer_template` -- this is to make sure chat templates are correctly set.

To evaluate internally at Ai2, just run `./shell_scripts/eval/eval_beaker.sh <model_name> <beaker dataset id>`.

### Analysis

Generally, analysis and utility scripts live in `scripts`. These are undocumented for now, but feel free to poke around in there. I also have a tonne of old scripts from a previous version of this project in `scripts/old`.

### Other selection methods

While RDS+ is our flagship method, we also tested a variety of other data selection methods in the paper and we provide code for you to reproduce/try them as well.

As mentioned above, the data selection pipeline can be viewed as indexing (scoring) -> selection -> training. Different methods prodvide differnet scoring, but the rest of the procedure is the same. Notes that some methods don't rely on validation set, therefore having slightly different pipeline.


**Embedding**

We support embedding models from [sentence-transformers](https://github.com/UKPLab/sentence-transformers) and HuggingFace. In the paper, we tested [NV-Embed-V2](https://huggingface.co/nvidia/NV-Embed-v2) and [GTR-t5-base](https://huggingface.co/sentence-transformers/gtr-t5-base).

To compute score with sentence embedding:

```bash
for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
    python -m minimal_multitask.compute_influence_sentence_embedding. \
        --seed 42 \
        --train_dataset data/training_data/tulu_v2_unfiltered_data_dedup.jsonl \
        --eval_dataset $dataset \
        --index_path sentence_embedding_based/gtrt5base_train_reps.pt \
        --save_dir sentence_embedding_based_${dataset}/ \
        --batch_size 1 \
        --model_name_or_path sentence-transformers/gtr-t5-base
done
```

**Perplexity**

We follow [CCDS](https://arxiv.org/abs/2410.16208) and support selection based on top-ppl (data points with highest perplexity) and mid-ppl (data points with middle perplexity).

To compute the perplexity on each training sample:

```bash
python -m minimal_multitask.compute_influence_perplexity \
    --model_name meta-llama/Llama-2-7b-hf \
    --seed 42 \
    --train_dataset data/training_data tulu_v2_unfiltered_data_dedup.jsonl \
    --save_dir perplexity/ \
    --batch_size 1 \
```

Then the selection can be done with (by default using top-ppl selection, specify --mid_ppl for mid_ppl selection):

```bash
python scripts/ppl_selection.py \
    --ppl_scores perplexity/nlls.pkl \
    --train_dataset data/training_data tulu_v2_unfiltered_data_dedup.jsonl \
    --output_size 10000 \
    --output_file_path perplexity/top_ppl_10k.json
```

**LESS**

We follow the procedure outlined in [the paper](https://arxiv.org/abs/2402.04333), using the [codebase shared by the authors](https://github.com/princeton-nlp/LESS).

**IFD**

We use the procedure outlined in [the paper](https://arxiv.org/abs/2308.12032), using [the codebase shared by the authors](https://github.com/tianyi-lab/Cherry_LLM).
We adjust the codebase slightly to accomodate multi-turn samples and custom chat templates.

**Random**

For `random-unbalanced` we simply randomly subsample with the `shuf` linux utility, e.g.:
```bash
sort --random-source=<(get_seeded_random 42) -R file.txt | head -k > output.jsonl
```
Replace `k` with the number of samples you wish to take. Credits to [this stackoverflow post](https://stackoverflow.com/questions/5914513/shuffling-lines-of-a-file-with-a-fixed-seed) for this snippet.

For `random-balanced`, we provide a script:
```bash
python scripts/construct_downsampled_balanced.py <input_file> <output_file> --seed 42 --num_samples k
```
Again, set `k` to the number of samples you wish to take. This script works by sampling uniformally from the list of values in the `dataset` field of the dataset.
If we run out of data from a given source, we divide its remaining `budget' equally among the dataset sources we have not yet exhausted.
