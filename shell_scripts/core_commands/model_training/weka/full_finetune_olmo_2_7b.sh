TRAIN_FILE=$1
EXP_NAME=$2

gantry run \
        --workspace hamishivi \
        --cluster ai2/allennlp-elara-cirrascale \
        --cluster ai2/saturn-cirrascale \
        --cluster ai2/jupiter-cirrascale-2 \
        --budget ai2/oe-adapt \
        --allow-dirty \
        --priority normal \
        --workspace ai2/minimal-multitask-finetuning \
        --gpus 1 \
        --env-secret HF_TOKEN=HF_TOKEN \
        --name $EXP_NAME \
        --pip requirements_olmo.txt \
        --weka=oe-adapt-default:/weka \
        --task-name $EXP_NAME \
        -- python -m minimal_multitask.instruction_tune \
                --model_name allenai/OLMo-2-1124-7B \
                --output_dir /results \
                --per_device_train_batch_size 1 \
                --gradient_accumulation_steps 128 \
                --num_train_epochs 2 \
                --learning_rate 2e-5 \
                --seed 42 \
                --bf16 \
                --warmup_ratio 0.03 \
                --lr_scheduler_type linear \
                --weight_decay 0. \
                --evaluation_strategy no \
                --save_strategy no \
                --logging_steps 1 \
                --is_llama=True \
                --use_hf_auth_token True \
                --is_olmo=True \
                --add_bos_token=True \
                --train_dataset $TRAIN_FILE