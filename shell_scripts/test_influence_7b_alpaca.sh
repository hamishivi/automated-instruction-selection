set -ex

for i in {2..5}; do
    # instruction tune model
    #python -m minimal_multitask.instruction_tune --model_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B --lora_rank 1 --save_dir results/alpaca_7b/lora_test_${i}
    # compute influences
    python -m minimal_multitask.compute_influence --model_name results/alpaca_7b/lora_test_${i} --top_k 10000 --instance_to_influences results/alpaca_7b/lora_test_top_1000_influences_${i}.pkl --seed $i --eval_dataset alpacafarm --index_path index_${i}.faiss
    # save top-k influences
    python -m minimal_multitask.get_top_influences --input_file results/alpaca_7b/lora_test_top_1000_influences_${i}.pkl --output_file results/alpaca_7b/lora_saved_instances_${i}.json
    # retrain model, but this time, we fully train it
    python -m minimal_multitask.instruction_tune  --model_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B --saved_instances results/alpaca_7b/lora_saved_instances_${i}.json --save_dir results/alpaca_7b/lora_influence_test_${i}
    # finally, evaluate.
    python -m minimal_multitask.run_alpaca_eval --model_name_or_path results/alpaca_7b/lora_influence_test_${i} --save_dir results/alpaca_7b/lora_alpacafarm_results_${i}
done


python -m minimal_multitask.compute_influence_flipped --model_name results/alpaca_7b/lora_64 --top_k 10000 --instance_to_influences results/alpaca_7b/lora_test_top_1000_influences_lora64.pkl --seed $i --eval_dataset alpacafarm