set -ex

for i in {1..5}; do
    # instruction tune model
    python -m minimal_multitask.instruction_tune --model_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B --lora_rank 1 --save_dir results/alpaca_7b/lora_test_${i} --use_flash_attention_2
    # compute influences
    python -m minimal_multitask.compute_influence --model_name results/alpaca_7b/lora_test_${i} --top_k 1000 --instance_to_influences results/alpaca_7b/lora_test_top_1000_influences_${i}.pkl --seed $i
    # save top-k influences
    python -m minimal_multitask.get_top_influences --input_file results/alpaca_7b/lora_test_top_1000_influences_${i}.pkl --output_file results/alpaca_7b/lora_saved_instances_${i}.json
    # retrain model, but this time, we fully train it
    python -m minimal_multitask.instruction_tune  --model_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B --saved_instances results/alpaca_7b/lora_saved_instances_${i}.json --save_dir results/alpaca_7b/lora_influence_test_${i} --use_flash_attention_2
    # finally, evaluate.
    python -m minimal_multitask.run_mmlu_eval --model_name_or_path results/alpaca_7b/lora_influence_test_${i} --save_dir results/alpaca_7b/lora_mmlu_results_${i} --use_chat_format --eval_batch_size 4 --load_in_8bit
    # performance just over the prompts we looked at influence for
    python -m minimal_multitask.performance_over_chosen_prompts --input_folder results/alpaca_7b/lora_mmlu_results_${i} &> results/alpaca_7b/influence_results_${i}.log
done
