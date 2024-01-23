set -ex

for i in {1..3}; do
    # save top-k influences
    python -m minimal_multitask.get_top_influences --input_file results/alpaca_7b/lora_test_top_1000_influences_${i}.pkl --output_file results/alpaca_7b/lora_saved_instances_filtered_1e3_${i}.json --filter_average_influence "-1e-3"
    # retrain model, but this time, we fully train it
    python -m minimal_multitask.instruction_tune  --model_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B --saved_instances results/alpaca_7b/lora_saved_instances_filtered_1e3_${i}.json --save_dir results/alpaca_7b/lora_influence_test_filtered_1e3_${i} 
    # finally, evaluate.
    python -m minimal_multitask.run_alpaca_eval --model_name_or_path results/alpaca_7b/lora_influence_test_${i} --save_dir results/alpaca_7b/lora_alpacafarm_results_filtered_1e3_${i}
done
