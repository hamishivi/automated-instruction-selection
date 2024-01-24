# Minimal multitask tuning repo

Very quick rundown: there is a lot of stuff in here from an older project, don't worry about it.

Look at the scripts in `shell_scripts` for how to run an influence experiment. Currently some paths are hardcoded for AI2 internal paths, but this is easily fixable.

Let me go through an MMLU example:

First, I set a seed value with `export i=1` or similar.

First, we can instruction tune with loras, and save the result. I have hardcoded some training hyperparameters, these may or may not be optimal.
```bash
python -m minimal_multitask.instruction_tune --model_name EleutherAI/pythia-70m --lora_rank 1 --save_dir results/pythia_70m/lora_test_${i}
```

Then, we can compute the influence. This method calculates all the train gradients and puts them into a FAISS index for reference, which can be saved to disk or just kept in memory. There is also `compute_influence_query_batching`, which instead computes the test gradients + hessian products, and then one by one calculates the train gradient and computes the influence, before saving. The latter is better if you are disk-space limited, but takes longer to re-run (since there is no saving of results).
```bash
python -m minimal_multitask.compute_influence_train_index --model_name results/pythia_70m/lora_test_${i} --top_k 1000 --instance_to_influences results/pythia_70m/lora_test_top_1000_influences_${i}.pkl --seed $i --eval_dataset mmlu --index_path index_${i}.faiss --save_index
```

Then, given a pickle with some saved influence scores, we extract the top-k. This script has some flags for adjusting what topk you want to retrieve.
```bash
# save top-k influences
python -m minimal_multitask.get_top_influences --input_file results/pythia_70m/lora_test_top_1000_influences_${i}.pkl --output_file results/pythia_70m/lora_saved_instances_${i}.json
```

We then fully-finetune on the chosen data.
```bash
# retrain model, but this time, we fully train it
python -m minimal_multitask.instruction_tune  --model_name EleutherAI/pythia-70m --saved_instances results/pythia_70m/lora_saved_instances_${i}.json --save_dir results/pythia_70m/lora_influence_test_${i} --use_fast_tokenizer
```

Finally, we evaluate on the task. For squad and alpacaEval, this is where we stop. For AlpacaEval, you'll need to set an OpenAI key.
```bash
# finally, evaluate.
python -m minimal_multitask.eval.run_mmlu_eval --model_name_or_path results/pythia_70m/lora_influence_test_${i} --save_dir results/pythia_70m/lora_mmlu_results_${i} --eval_batch_size 4 --use_chat_format
```

For MMLU, we can also just examine the prompts used for selecting influence.
```bash
# performance just over the prompts we looked at influence for
python -m minimal_multitask.performance_over_chosen_prompts --input_folder results/pythia_70m/lora_mmlu_results_${i}
```

And thats it! Squad and AlpacaEval work roughly the same way.
