set -ex

BEAKER_PATH=$2
# SUBPATH=$3

# command for gantry here
# includes oai key and turning off alpaca eval 2 for alpaca eval stuff.
GANTRY_CMD="gantry run --cluster ai2/allennlp-cirrascale --cluster ai2/general-cirrascale --cluster ai2/general-cirrascale-a100-80g-ib --cluster ai2/mosaic-cirrascale-a100 --cluster ai2/pluto-cirrascale --cluster ai2/s2-cirrascale-l40 --budget ai2/oe-adapt --allow-dirty --priority preemptible --workspace ai2/minimal-multitask-finetuning --gpus 1 --env-secret OPENAI_API_KEY=OPENAI_API_KEY --env LD_LIBRARY_PATH=/opt/conda/envs/venv/lib --env IS_ALPACA_EVAL_2=False --dataset ${BEAKER_PATH}:/model"

for file in /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered/subshards/*.jsonl; do
    if [ -f "$file" ]; then
        shard=$(basename $file .jsonl)
        echo "Processing $shard"
        $GANTRY_CMD --name logix_alpacafarm_10k_rand --task-name logix_alpacafarm_10k_rand -- accelerate launch \
            --mixed_precision bf16 \
            -m minimal_multitask.compute_influence_logix \
            --use_flash_attention_2 \
            --gradient_checkpointing \
            --train_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered/subshards/${shard}.jsonl \
            --eval_dataset alpacafarm \
            --grad_save_path tmp/tulu_unfiltered_logix_tulu_unfiltered_${shard} \
            --instance_to_influences tmp/alpacafarm_tulu_unfiltered_logix_tulu_unfiltered_${shard}.pkl
        exit
    fi
done