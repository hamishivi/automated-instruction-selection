# alpacafarm, mmlu_shots already done.
for eval_dataset in alpaca_mmlu_shots285_cossim_min10000 alpaca_squad500_cossim_min10000
do
    gantry run --workspace hamishivi --cluster ai2/allennlp-cirrascale --budget ai2/oe-adapt --allow-dirty --priority normal --workspace ai2/minimal-multitask-finetuning --gpus 1 --env-secret HF_TOKEN=HF_TOKEN --name ${eval_dataset}_cossim_max10000 -- python -m minimal_multitask.instruction_tune \
        --model_name meta-llama/Llama-2-7b-hf \
        --output_dir /results \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 128 \
        --num_train_epochs 2 \
        --learning_rate 2e-5 \
        --seed 42 \
        --evaluation_strategy no \
        --save_strategy no \
        --logging_steps 1 \
        --is_llama=True \
        --use_hf_auth_token True \
        --train tulu2 \
        --saved_instances /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/cosine_min/${eval_dataset}.json
done

# gantry run --workspace hamishivi --cluster ai2/allennlp-cirrascale --budget ai2/oe-adapt --allow-dirty --priority normal --workspace ai2/minimal-multitask-finetuning --gpus 1 --env-secret HF_TOKEN=HF_TOKEN --name overall_tulu2_min_min --task-name overall_tulu2_min_min -- python -m minimal_multitask.instruction_tune \
#         --model_name meta-llama/Llama-2-7b-hf \
#         --output_dir /results \
#         --per_device_train_batch_size 1 \
#         --gradient_accumulation_steps 128 \
#         --num_train_epochs 2 \
#         --learning_rate 2e-5 \
#         --seed 42 \
#         --evaluation_strategy no \
#         --save_strategy no \
#         --logging_steps 1 \
#         --is_llama=True \
#         --use_hf_auth_token True \
#         --train tulu2 \
#         --saved_instances /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/results/llama_2_128lora_tulu2/overall_min_min.json