set -ex

export OPENAI_API_KEY=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

# instruction tune model
python -m minimal_multitask.instruction_tune \
    --model_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B \
    --lora_rank 128 \
    --output_dir results/alpaca_less/lora_rank_128 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --seed 42
# compute influences
python -m minimal_multitask.compute_influence_train_index \
    --model_name results/alpaca_less/lora_rank_128_ \
    --top_k 52002 \
    --instance_to_influences results/alpaca_less/lora_128_results_alpacafarm.pkl \
    --seed 42 \
    --random_transform 8192 \
    --normalise_influences \
    --eval_dataset alpacafarm \
    --index_path index_lora_128_norm.faiss \
    --vanilla_gradients
# save top-k influences
python -m minimal_multitask.get_top_influences \
    --input_file results/alpaca_less/lora_128_results_alpacafarm.pkl \
    --output_file results/alpaca_less/lora_128_instances_alpacafarm.json --take_top_k 250
# retrain model, but this time, we fully train it
python -m minimal_multitask.instruction_tune \
    --model_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B \
    --saved_instances results/alpaca_less/lora_128_instances_alpacafarm_hvp10.json \
    --output_dir results/alpaca_less/alpaca_less_lora_128_influence_ff_af_hvp10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --seed 42
# finally, evaluate.
python -m minimal_multitask.run_alpaca_eval \
    --model_name_or_path results/alpaca_less/alpaca_less_lora_128_influence_ff_af \
    --save_dir results/alpaca_less/alpaca_less_lora_128_influence_ff_af_results
