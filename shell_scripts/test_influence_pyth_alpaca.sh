set -ex

for i in {2..5}; do
    # instruction tune model
    python -m minimal_multitask.instruction_tune --model_name EleutherAI/pythia-70m --save_dir results/pythia_70m/alpaca/lora_test_${i} --use_fast_tokenizer
    # compute influences
    python -m minimal_multitask.compute_influence --model_name results/pythia_70m/alpaca/lora_test__${i} --top_k 10000 --instance_to_influences results/pythia_70m/alpaca/lora_test_top_1000_influences_${i}.pkl --seed $i --eval_dataset alpacafarm --index_path index_${i}.faiss
    # save top-k influences
    python -m minimal_multitask.get_top_influences --input_file results/pythia_70m/alpaca/lora_test_top_1000_influences_$${i}.pkl --output_file results/pythia_70m/alpaca/lora_saved_instances_${i}.json --topk 1000
    # retrain model, but this time, we fully train it
    python -m minimal_multitask.instruction_tune  --model_name EleutherAI/pythia-70m --saved_instances results/pythia_70m/alpaca/lora_saved_instances_${i}.json --save_dir results/pythia_70m/alpaca/lora_influence_test_${i} --use_fast_tokenizer
    # finally, evaluate.
    python -m minimal_multitask.run_alpaca_eval --model_name_or_path results/pythia_70m/alpaca/lora_influence_test_${i} --save_dir results/pythia_70m/alpaca/lora_alpacafarm_results_${i}
done

# random selection
python -m minimal_multitask.instruction_tune  --model_name EleutherAI/pythia-70m --random_select 20103 --save_dir results/pythia_70m/alpaca/random_${i} --use_fast_tokenizer