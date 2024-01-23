# instruction tune model
# python -m minimal_multitask.instruction_tune --model_name EleutherAI/pythia-70m  --save_dir jlt_test
# compute influences
python -m minimal_multitask.compute_influence --model_name jlt_test --use_fjlt --top_k 1000 --instance_to_influences jlt_test_top_1000_influences.pkl
# save top-k influences
python -m minimal_multitask.get_top_influences --input_file jlt_test_top_1000_influences.pkl --output_file jlt_saved_instances.json
# retrain model, but this time, we fully train it
python -m minimal_multitask.instruction_tune  --model_name EleutherAI/pythia-70m --saved_instances jlt_test_top_1000_influences.json --save_dir jlt_influence_test
# finally, evaluate.
python -m minimal_multitask.run_mmlu_eval --model_name_or_path jlt_influence_test
