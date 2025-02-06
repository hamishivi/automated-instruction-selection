TRAIN_FILE=$1
EXP_NAME=$2

# gantry run \
#     --workspace hamishivi \
#     --cluster ai2/allennlp-cirrascale \
#     --cluster ai2/pluto-cirrascale \
#     --budget ai2/oe-adapt \
#     --allow-dirty \
#     --priority normal \
#     --workspace ai2/minimal-multitask-finetuning \
#     --gpus 1 \
#     --env-secret HF_TOKEN=HF_TOKEN \
#     --name $EXP_NAME \
#     --task-name $EXP_NAME \
#     -- python -m minimal_multitask.instruction_tune \
#             --model_name meta-llama/Llama-2-7b-hf \
#             --output_dir /results \
#             --per_device_train_batch_size 1 \
#             --gradient_accumulation_steps 128 \
#             --num_train_epochs 2 \
#             --learning_rate 2e-5 \
#             --seed 42 \
#             --warmup_ratio 0.03 \
#             --lr_scheduler_type linear \
#             --weight_decay 0. \
#             --evaluation_strategy no \
#             --save_strategy no \
#             --logging_steps 1 \
#             --is_llama=True \
#             --use_hf_auth_token True \
#             --train $TRAIN_FILE

# for dataset in alpacafarm gsm8k_shots bbh_shots tydiqa_shots codex squad mmlu_shots; do
BEAKER_TOKEN='V3xbE/+01bpq1vtR'
export BEAKER_TOKEN=$BEAKER_TOKEN
for seed in 42; do
    for datasize in 10000; do
        TRAIN_FILE=/weka/oe-adapt-default/muruz/influence-tuning/selections/tulu2_unfiltered_subsample/balance_random_${datasize}_seed${seed}.json
        EXP_NAME=balance_random_${datasize}_seed${seed}
        gantry run \
            --cluster ai2/saturn-cirrascale \
            --cluster ai2/allennlp-elara-cirrascale \
            --budget ai2/oe-adapt \
            --priority normal \
            --workspace ai2/minimal-multitask-finetuning \
            --allow-dirty \
            --gpus 1 \
            --env-secret HF_TOKEN=HF_TOKEN \
            --name $EXP_NAME \
            --task-name $EXP_NAME \
            --weka oe-adapt-default:/weka/oe-adapt-default \
            -- python -m minimal_multitask.instruction_tune \
                    --model_name meta-llama/Llama-2-7b-hf \
                    --output_dir /results \
                    --per_device_train_batch_size 1 \
                    --gradient_accumulation_steps 128 \
                    --num_train_epochs 2 \
                    --learning_rate 2e-5 \
                    --seed 42 \
                    --warmup_ratio 0.03 \
                    --lr_scheduler_type linear \
                    --weight_decay 0. \
                    --evaluation_strategy no \
                    --save_strategy no \
                    --logging_steps 1 \
                    --is_llama=True \
                    --use_hf_auth_token True \
                    --train $TRAIN_FILE
    done
done



for seed in 42; do
    for datasize in 10000; do
        TRAIN_FILE=/weka/oe-adapt-default/muruz/influence-tuning/selections/tulu2_unfiltered_subsample/balance_random_${datasize}_seed${seed}.json
        EXP_NAME=balance_random_${datasize}_seed${seed}
        python -m minimal_multitask.instruction_tune \
                    --model_name meta-llama/Llama-2-7b-hf \
                    --output_dir /results \
                    --per_device_train_batch_size 1 \
                    --gradient_accumulation_steps 128 \
                    --num_train_epochs 2 \
                    --learning_rate 2e-5 \
                    --seed 42 \
                    --warmup_ratio 0.03 \
                    --lr_scheduler_type linear \
                    --weight_decay 0. \
                    --evaluation_strategy no \
                    --save_strategy no \
                    --logging_steps 1 \
                    --is_llama=True \
                    --use_hf_auth_token True \
                    --train $TRAIN_FILE
    done
done