TRAIN_FILE=$1
EXP_NAME=$2

gantry run \
        --workspace hamishivi \
        --cluster ai2/allennlp-elara-cirrascale \
        --cluster ai2/saturn-cirrascale \
        --cluser ai2/neptune-cirrascale \
        --budget ai2/oe-adapt \
        --allow-dirty \
        --priority normal \
        --workspace ai2/minimal-multitask-finetuning \
        --gpus 2 \
        --env-secret HF_TOKEN=HF_TOKEN \
        --name $EXP_NAME \
        --weka=oe-adapt-defult:/weka \
        --task-name $EXP_NAME \
        -- accelerate launch \
                --mixed_precision bf16 \
                --use_deepspeed \
                --deepspeed_config_file ds_config.json -m minimal_multitask.instruction_tune \
                --model_name meta-llama/Llama-3.1-8B \
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