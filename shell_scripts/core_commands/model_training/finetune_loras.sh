TRAIN_FILE=$1
EXP_NAME=$2
BEAKER_MODEL_ID=$3

gantry run \
        --workspace hamishivi \
        --cluster ai2/allennlp-cirrascale \
        --cluster ai2/pluto-cirrascale \
        --budget ai2/oe-adapt \
        --allow-dirty \
        --priority normal \
        --dataset "${BEAKER_MODEL_ID}:/model" \
        --workspace ai2/minimal-multitask-finetuning \
        --gpus 1 --env-secret HF_TOKEN=HF_TOKEN \
        --name $EXP_NAME \
        --task-name $EXP_NAME \
        -- python -m minimal_multitask.instruction_tune \
                --model_name /model \
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
                --lora_alpha 512 \
                --lora_rank 128 \
                --use_hf_auth_token True \
                --train $TRAIN_FILE