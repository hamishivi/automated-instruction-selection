set -ex

# TODO: I can make this bash script more generalist

experiment_name="lora_1_squad"
lora_rank=1
eval_dataset="squad"

for i in {2..5}; do
    # instruction tune model if we haven't already
    if [ ! -d results/alpaca_7b/lora_test_${i} ]; then
        echo "model not found, training model"
        python -m minimal_multitask.instruction_tune --model_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B --lora_rank $lora_rank --save_dir results/alpaca_7b/lora_test_${i}
    fi
    # compute influences
    python -m minimal_multitask.compute_influence_train_index --model_name results/alpaca_7b/lora_test_${i} --top_k 10000 --instance_to_influences results/alpaca_7b/lora_test_top_1000_influences_${i}.pkl --seed $i --eval_dataset $eval_dataset --index_path index_${i}.faiss
    # save top-k influences
    python -m minimal_multitask.get_top_influences --input_file results/alpaca_7b/lora_test_top_1000_influences_${i}.pkl --output_file results/alpaca_7b/lora_saved_instances_${i}.json
    # retrain model, but this time, we fully train it
    python -m minimal_multitask.instruction_tune  --model_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B --saved_instances results/alpaca_7b/lora_saved_instances_${i}.json --save_dir results/alpaca_7b/lora_influence_test_${i}
    # finally, evaluate.
    python -m minimal_multitask.eval.run_squad_eval --model_name results/alpaca_7b/lora_influence_test_${i}
done

# experiment to run once index is built
python -m minimal_multitask.compute_influence_vanilla_gradient_train_index --model_name results/alpaca_7b/lora_test_${i} --top_k 10000 --instance_to_influences results/alpaca_7b/lora_test_top_1000_influences_vg_${i}.pkl --seed $i --eval_dataset $eval_dataset --index_path index_${i}.faiss