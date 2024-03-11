# eval command for a given model

MODEL_PATH=/net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B

# command for gantry here
# includes oai key and turning off alpaca eval 2 for alpaca eval stuff.
GANTRY_CMD="gantry run --cluster ai2/allennlp-cirrascale --cluster ai2/general-cirrascale --cluster ai2/general-cirrascale-a100-80g-ib --cluster ai2/mosaic-cirrascale-a100 --cluster ai2/s2-cirrascale-l40 --budget ai2/oe-adapt --allow-dirty --priority preemptible --workspace ai2/minimal-multitask-finetuning --gpus 1 --env-secret OPENAI_API_KEY=OPENAI_API_KEY --env IS_ALPACA_EVAL_2=False --"

# mmlu
# 0shot eval
# NB: bsz 1 due to a bug in the tokenizer for llama 2.
$GANTRY_CMD python -m minimal_multitask.eval.mmlu.run_mmlu_eval \
    --ntrain 0 \
    --save_dir ${MODEL_PATH}/mmlu_results \
    --model_name_or_path ${MODEL_PATH} \
    --eval_batch_size 1 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format

# gsm8k
# cot, 8 shot.
$GANTRY_CMD python -m minimal_multitask.eval.gsm.run_eval \
    --data_dir /net/nfs.cirrascale/allennlp/yizhongw/open-instruct/data/eval/gsm/ \
    --max_num_examples 200 \
    --save_dir ${MODEL_PATH}/gsm_results \
    --model_name_or_path ${MODEL_PATH} \
    --n_shot 8 \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm

# bbh
# cot, 3-shot
$GANTRY_CMD python -m minimal_multitask.eval.bbh.run_eval \
    --data_dir /net/nfs.cirrascale/allennlp/yizhongw/open-instruct/data/eval/bbh \
    --save_dir ${MODEL_PATH}/bbh_results \
    --model_name_or_path ${MODEL_PATH} \
    --max_num_examples_per_task 40 \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format

# tydiqa
# 1-shot, with context
$GANTRY_CMD python -m minimal_multitask.eval.tydiqa.run_eval \
    --data_dir /net/nfs.cirrascale/allennlp/yizhongw/open-instruct/data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir ${MODEL_PATH}/tydiqa_results \
    --model_name_or_path ${MODEL_PATH} \
    --eval_batch_size 20 \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format

# codex
# humaneval pack, using my test split.
# measure p@10
$GANTRY_CMD python -m minimal_multitask.eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
    --data_file_hep data/eval/codex_humaneval/humanevalpack.jsonl  \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format \
    --eval_pass_at_ks 10 \
    --unbiased_sampling_size_n 10 \
    --temperature 0.8 \
    --save_dir ${MODEL_PATH}/codex_results \
    --model_name_or_path ${MODEL_PATH} \
    --use_vllm

# squad
# just eval as normal
$GANTRY_CMD python -m minimal_multitask.eval.squad.run_squad_eval \
    --model_name ${MODEL_PATH} \
    --output_file ${MODEL_PATH}/squad_results

# alpaca eval
# use my test split
$GANTRY_CMD python -m minimal_multitask.eval.alpaca_eval.run_alpaca_eval \
    --save_dir ${MODEL_PATH}/alpacaeval_results \
    --model_name_or_path ${MODEL_PATH}

echo "Done evaluation!"