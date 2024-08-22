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

# base select
gantry run --workspace hamishivi --cluster ai2/allennlp-cirrascale --budget ai2/oe-adapt --allow-dirty --priority normal --workspace ai2/minimal-multitask-finetuning --gpus 1 --env-secret HF_TOKEN=HF_TOKEN  --name logix_rank48_cosine_hessian_squad --task-name logix_rank48_cosine_hessian_squad -- python -m minimal_multitask.instruction_tune \
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
        --train /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/selection_files/logix_rank48_cosine_hessian/squad/top_10k.json

# random select lora finetune
gantry run --workspace hamishivi --cluster ai2/allennlp-cirrascale --budget ai2/oe-adapt --allow-dirty --priority normal --workspace ai2/minimal-multitask-finetuning --gpus 1 --env-secret HF_TOKEN=HF_TOKEN  --name lora_and_ff_10k_rand --task-name lora_and_ff_10k_rand -- python -m minimal_multitask.instruction_tune \
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
        --lora_alpha 512 \
        --lora_rank 128 \
        --use_hf_auth_token True \
        --train /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered/tulu_v2_unfiltered_data.jsonl \
        --random_select 10000 \
        --lora_ff_train true



gantry run --workspace hamishivi --cluster ai2/allennlp-cirrascale --budget ai2/oe-adapt --allow-dirty --priority normal --dataset '01J0RNB6D6M80FN67NDH17KE54:/model' --workspace ai2/minimal-multitask-finetuning --gpus 1 --env-secret HF_TOKEN=HF_TOKEN  --name less_warmup_hessian_squad --task-name less_warmup_hessian_squad -- python -m minimal_multitask.instruction_tune \
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
        --use_hf_auth_token True \
        --train /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/less_warmup_hessian/squad/top_10k.json

# lora version for grads
gantry run --workspace hamishivi --cluster ai2/allennlp-cirrascale --budget ai2/oe-adapt --allow-dirty --priority normal --dataset '01J0RNB6D6M80FN67NDH17KE54:/model' --workspace ai2/minimal-multitask-finetuning --gpus 1 --env-secret HF_TOKEN=HF_TOKEN --name random_unfil_train_then_loras --task-name random_unfil_train_then_loras -- python -m minimal_multitask.instruction_tune \
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
        --train /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered/tulu_v2_unfiltered_data_300k_sample.jsonl

# random 10k 4 epochs, both unfiltered and filtered
gantry run --workspace hamishivi --cluster ai2/allennlp-cirrascale --budget ai2/oe-adapt --allow-dirty --priority normal --workspace ai2/minimal-multitask-finetuning --dataset '01J0RNB6D6M80FN67NDH17KE54:/model' --gpus 1 --env-secret HF_TOKEN=HF_TOKEN  --name random_train_then_sel_aeval --task-name random_train_then_sel_aeval -- python -m minimal_multitask.instruction_tune \
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
        --use_hf_auth_token True \
        --train /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/logix_aeval_rand10k_sel/alpacafarm_top_10k.json