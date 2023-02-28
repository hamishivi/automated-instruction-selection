
WORKSPACE=ai2/minimal-multitask-finetuning
CLUSTER=ai2/allennlp-cirrascale

# generic gantry command for this.
while read line; do
      arr=($line)
      og_name=${arr[0]}
      dataset_id=${arr[1]}
      chkpt=$(beaker dataset ls ${dataset_id} | grep -oPe checkpoint\-[0-9]+ | tail -n 1)
      # echo "${og_name} ${dataset_id} ${chkpt}"
      gantry run -y -n ${og_name}-evaluate --allow-dirty \
        --workspace $WORKSPACE \
        --gpus 1 \
        --nfs \
        --dataset ${dataset_id}:/inputs \
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
        --dry-run \
        --save-spec tmp.yaml \
        -- python minimal_multitask/train.py --do_eval --output_dir /results --eval_tasks evaluation --bf16 --model_name /inputs/  --predict_with_generate --metrics_output /results/metrics.json --generation_max_length 256 --train_tasks glue_mrpc_generate --tokenizer_name t5-base
      sed -i 's/\- mountPath\: \/inputs/\- mountPath\: \/inputs\n    subPath\: checkpoint-1950/g' tmp.yaml

      beaker experiment create tmp.yaml -n ${og_name}-evaluate -p preemptible -w $WORKSPACE
done < data/eval_task_datasets.txt

rm tmp.yaml
