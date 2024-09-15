# eval command for a given model
# runs evaluation on beaker. each eval script is run in a separate job.
set -ex

MODEL_NAME=$1
BEAKER_PATH=$2

# command for gantry here
# includes oai key and turning off alpaca eval 2 for alpaca eval stuff.
# dataset 01J6QSXVDS4MN0W45HB2MHWXQN is the data for the eval
GANTRY_CMD="gantry run --cluster ai2/allennlp-cirrascale --cluster ai2/general-cirrascale --cluster ai2/mosaic-cirrascale-a100 --cluster ai2/pluto-cirrascale --cluster ai2/s2-cirrascale-l40 --cluster ai2/jupiter-cirrascale-2 --no-nfs --budget ai2/oe-adapt --allow-dirty --priority preemptible --workspace ai2/minimal-multitask-finetuning --gpus 1 --env-secret OPENAI_API_KEY=OPENAI_API_KEY --env IS_ALPACA_EVAL_2=False --dataset ${BEAKER_PATH}:/model --dataset 01J6QSXVDS4MN0W45HB2MHWXQN:/data"

# mmlu
# 0shot eval
# NB: bsz 1 due to a bug in the tokenizer for llama 2.
$GANTRY_CMD --name ${MODEL_NAME}_mmlu_0shot -- python -m minimal_multitask.eval.mmlu.run_mmlu_eval \
    --ntrain 0 \
    --data_dir /data/eval/mmlu/ \
    --save_dir /results \
    --model_name_or_path /model \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format

# gsm8k
# cot, 8 shot.
$GANTRY_CMD --name ${MODEL_NAME}_gsm_cot_full -- python -m minimal_multitask.eval.gsm.run_eval \
    --data_dir /data/eval/gsm/ \
    --save_dir /results \
    --model_name_or_path /model \
    --n_shot 8 \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm

# bbh
# cot, 3-shot
$GANTRY_CMD --name ${MODEL_NAME}_bbh_cot_full -- python -m minimal_multitask.eval.bbh.run_eval \
    --data_dir /data/eval/bbh \
    --save_dir /results \
    --model_name_or_path /model \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format

# tydiqa
# 1-shot, with context
$GANTRY_CMD --name ${MODEL_NAME}_tydiqa_goldp_full -- python -m minimal_multitask.eval.tydiqa.run_eval \
    --data_dir /data/eval/tydiqa/ \
    --n_shot 1 \
    --max_context_length 512 \
    --save_dir /results \
    --model_name_or_path /model \
    --eval_batch_size 20 \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format

# codex
# humaneval pack, using my test split.
# measure p@10
$GANTRY_CMD --name ${MODEL_NAME}_codex_pass10 -- python -m minimal_multitask.eval.codex_humaneval.run_eval \
    --data_file /data/eval/codex_humaneval/HumanEval.jsonl.gz  \
    --data_file_hep /data/eval/codex_humaneval/humanevalpack.jsonl  \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format \
    --eval_pass_at_ks 10 \
    --unbiased_sampling_size_n 10 \
    --temperature 0.8 \
    --save_dir /results \
    --model_name_or_path /model \
    --use_vllm

# squad
# just eval as normal
$GANTRY_CMD --name ${MODEL_NAME}_squad_context_fixed -- python -m minimal_multitask.eval.squad.run_squad_eval \
    --model_name_or_path /model \
    --output_file "/results/predictions.json" \
    --metrics_file "/results/metrics.json" \
    --generation_file "/results/generation.json" \
    --use_vllm

# alpaca eval
# use my test split
$GANTRY_CMD --name ${MODEL_NAME}_alpaca_eval -- python -m minimal_multitask.eval.alpaca_eval.run_alpaca_eval \
    --save_dir /results \
    --model_name_or_path /model \
    --use_vllm

echo "Done evaluation!"