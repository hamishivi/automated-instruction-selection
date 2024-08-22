

BEAKER_PATH=$1
# SUBPATH=$3

# command for gantry here
# includes oai key and turning off alpaca eval 2 for alpaca eval stuff.
GANTRY_CMD="gantry run --cluster ai2/allennlp-cirrascale --budget ai2/oe-adapt --allow-dirty --priority normal --workspace ai2/minimal-multitask-finetuning --gpus 1 --env-secret OPENAI_API_KEY=OPENAI_API_KEY --env LD_LIBRARY_PATH=/opt/conda/envs/venv/lib --env IS_ALPACA_EVAL_2=False --dataset ${BEAKER_PATH}:/model"

for file in /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered/subshards/*.jsonl; do
    if [ -f "$file" ]; then
        shard=$(basename $file .jsonl)
            echo "Processing $shard"
            $GANTRY_CMD --name rand_10k_logix_rank48_hesraw_fp32_${shard} --task-name rand_10k_logix_rank48_hesraw_fp32_${shard} -- accelerate launch \
                --mixed_precision bf16 \
                -m minimal_multitask.compute_influence_logix \
                --model_name /model \
                --use_flash_attention_2 \
                --gradient_checkpointing \
                --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered/subshards/${shard}.jsonl \
                --eval_dataset alpacafarm \
                --hessian_type raw \
                --beaker \
                --logra_rank 48 \
                --grad_save_path /results/tulu_unfiltered_logix_tulu_unfiltered_${shard} \
                --instance_to_influences /results/alpacafarm_tulu_unfiltered_logix_tulu_unfiltered_${shard}.pkl \
                --logra_precision float32
    fi
done

# test 

# accelerate launch \
#     --mixed_precision bf16 \
#     -m minimal_multitask.compute_influence_logix \
#     --model_name ff_random_10k \
#     --use_flash_attention_2 \
#     --gradient_checkpointing \
#     --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered/tulu_v2_unfiltered_data_1k_sample.jsonl \
#     --eval_dataset alpacafarm \
#     --grad_save_path tmp/grad_save \
#     --instance_to_influences tmp/influences.pkl \
#     --logra_rank 6 \
#     --hessian_type raw \
#     --logra_precision float32