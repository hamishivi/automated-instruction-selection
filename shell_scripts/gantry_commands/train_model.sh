export EXP_NAME=tulu_2_10k_random_s1

gantry run --workspace ai2/minimal-multitask-finetuning --cluster ai2/allennlp-cirrascale --budget ai2/oe-adapt --allow-dirty --priority normal --env-secret HF_TOKEN=HF_TOKEN --gpus 1 --name $EXP_NAME --task-name $EXP_NAME -- python -m minimal_multitask.instruction_tune \
    --model_name meta-llama/Llama-2-7b-hf \
    --output_dir /results \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --seed 1 \
    --use_hf_auth_token True \
    --save_strategy no \
    --evaluation_strategy no \
    --logging_steps 1 \
    --is_llama=True \
    --train tulu2 \
    --random_select 10000
