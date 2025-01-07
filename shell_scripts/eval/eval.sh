# eval command for a given model
set -ex

MODEL_PATH=$1

# mmlu
# 0shot eval
# NB: bsz 1 due to a bug in the tokenizer for llama 2.
python -m minimal_multitask.eval.mmlu.run_mmlu_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu/ \
    --save_dir results_mmlu \
    --model_name_or_path ${MODEL_PATH} \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format

# gsm8k
# cot, 8 shot.
ython -m minimal_multitask.eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --save_dir results_gsm8k \
    --model_name_or_path ${MODEL_PATH} \
    --n_shot 8 \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm

# bbh
# cot, 3-shot
python -m minimal_multitask.eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results_bbh \
    --model_name_or_path ${MODEL_PATH} \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format

# tydiqa
# 1-shot, with context
python -m minimal_multitask.eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_context_length 512 \
    --save_dir results_tydiqa \
    --model_name_or_path ${MODEL_PATH} \
    --eval_batch_size 20 \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format

# codex
# humaneval pack, using my test split.
# measure p@10
python -m minimal_multitask.eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
    --data_file_hep data/eval/codex_humaneval/humanevalpack.jsonl  \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format \
    --eval_pass_at_ks 10 \
    --unbiased_sampling_size_n 10 \
    --temperature 0.8 \
    --save_dir results_humaneval \
    --model_name_or_path ${MODEL_PATH} \
    --use_vllm

# squad
# just eval as normal
mkdir -p results_squad
python -m minimal_multitask.eval.squad.run_squad_eval \
    --model_name_or_path ${MODEL_PATH} \
    --output_file "results_squad/predictions.json" \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format \
    --metrics_file "results_squad/metrics.json" \
    --generation_file "results_squad/generation.json" \
    --use_vllm

# alpaca eval
# use my test split
python -m minimal_multitask.eval.alpaca_eval.run_alpaca_eval \
    --save_dir results_alpaca_eval \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format \
    --model_name_or_path ${MODEL_PATH} \
    --use_vllm

echo "Done evaluation!"
