set -ex
# afarm already done, mmlu usually takes hours.
for eval_dataset in gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots
do
    python -m minimal_multitask.get_top_influences \
        --input_file results/llama_2_128lora_tulu2/${eval_dataset}.pkl \
        --output_file results/llama_2_128lora_tulu2/top_10k_${eval_dataset}_top_min.json \
        --output_size 10000 \
        --selection_method min \
        --train_dataset tulu2
done