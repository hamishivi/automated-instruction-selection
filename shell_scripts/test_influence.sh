
for i in {1..5}; do
    # instruction tune model
    # python -m minimal_multitask.instruction_tune --model_name EleutherAI/pythia-70m --lora_rank 1 --save_dir results/pythia_70m/lora_test_${i}
    # compute influences
    python -m minimal_multitask.compute_influence --model_name results/pythia_70m/lora_test_${i} --top_k 1000 --instance_to_influences results/pythia_70m/lora_test_top_1000_influences_${i}.pkl --seed $i --eval_dataset mmlu
    # save top-k influences
    python -m minimal_multitask.get_top_influences --input_file results/pythia_70m/lora_test_top_1000_influences_${i}.pkl --output_file results/pythia_70m/lora_saved_instances_${i}.json
    # retrain model, but this time, we fully train it
    python -m minimal_multitask.instruction_tune  --model_name EleutherAI/pythia-70m --saved_instances results/pythia_70m/lora_saved_instances_${i}.json --save_dir results/pythia_70m/lora_influence_test_${i} --use_fast_tokenizer
    # finally, evaluate.
    python -m minimal_multitask.run_mmlu_eval --model_name_or_path results/pythia_70m/lora_influence_test_${i} --save_dir results/pythia_70m/lora_mmlu_results_${i} --eval_batch_size 4 --use_chat_format
    # performance just over the prompts we looked at influence for
    python -m minimal_multitask.performance_over_chosen_prompts --input_folder results/pythia_70m/lora_mmlu_results_${i} &> results/pythia_70m/my_lora_results_seed${i}.log
done