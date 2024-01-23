set -ex

for i in {1..1}; do
    # retrain model, but this time, we fully train it
    python -m minimal_multitask.instruction_tune  --model_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B --random_select 25000 --save_dir results/alpaca_7b/random_25k_${i} --seed $i
    # evaluate
    python -m minimal_multitask.run_alpaca_eval --model_name_or_path results/alpaca_7b/random_25k_${i} --save_dir results/alpaca_7b/lora_alpacafarm_results_random25k_${i} &> results/alpaca_7b/random_results_${i}.log
    # cleanup
    rm -rf results/alpaca_7b/random_25k_${i}
done
