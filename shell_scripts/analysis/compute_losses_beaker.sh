set -ex

MODEL_NAME=$1
BEAKER_PATH=$2


gantry run \
    --cluster ai2/allennlp-cirrascale \
    --cluster ai2/general-cirrascale \
    --cluster ai2/s2-cirrascale-l40 \
    --budget ai2/oe-adapt \
    --allow-dirty \
    --priority normal \
    --workspace ai2/minimal-multitask-finetuning \
    --gpus 1 \
    --env-secret OPENAI_API_KEY=OPENAI_API_KEY \
    --env IS_ALPACA_EVAL_2=False \
    --env-secret HF_TOKEN=HF_TOKEN \
    --dataset ${BEAKER_PATH}:/model \
    --name compute_test_loss_${MODEL_NAME} \
    --task-name compute_test_loss_${MODEL_NAME} \
    -- python -m minimal_multitask.compute_test_loss --model_name /model --tokenizer meta-llama/Llama-2-7b-hf --eval_datasets mmlu_shots gsm8k_shots bbh_shots tydiqa_shots codex squad alpacafarm mmlu gsm8k bbh tydiqa codex_test alpacafarm_test squad_test --seed 42 --results /results/metrics.json
