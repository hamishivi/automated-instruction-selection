
WORKSPACE=ai2/minimal-multitask-finetuning
CLUSTER=ai2/allennlp-cirrascale

# generic gantry command for this.
while read task; do
      gantry run -y -n "${task// /_}"-pair --allow-dirty \
        --workspace $WORKSPACE \
        --gpus 1 \
        --nfs \
        --priority preemptible \
        --cluster ai2/allennlp-cirrascale \
        --cluster ai2/mosaic-cirrascale-a100 \
        --cluster ai2/general-cirrascale-a100-80g-ib \
        --env 'HF_HOME=/net/nfs.cirrascale/allennlp/hamishi/.hf' \
        --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
        --env 'HF_DATASETS_OFFLINE=1' \
        --env 'TRANSFORMERS_OFFLINE=1' \
        --beaker-image 'ai2/pytorch1.13.0-cuda11.6-python3.9' \
        --venv 'base' \
        --pip requirements.txt \
        -- python minimal_multitask/train.py --do_train --do_eval --output_dir /results --train_tasks $task --eval_tasks $task --evaluation_strategy no --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --learning_rate 0.00005 --num_train_epochs 5 --bf16 --model_name google/t5-xl-lm-adapt --predict_with_generate --metrics_output /results/metrics.json --generation_max_length 256 --save_total_limit 2 --save_strategy epoch
done < data/task_pairs.txt


