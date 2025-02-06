set -ex

eval_datasets=(mmlu_shots gsm8k_shots bbh_shots tydiqa_shots codex squad alpacafarm)
test_datasets=(mmlu gsm8k bbh tydiqa codex_test alpacafarm_test squad_test)

for ds in ${test_datasets[@]}; do
    python -m minimal_multitask.compute_test_loss --model_name meta-llama/Llama-2-7b-hf --eval_dataset $ds --seed 42
done


# beaker version that does everything
