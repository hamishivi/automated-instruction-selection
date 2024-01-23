
WORKSPACE=ai2/minimal-multitask-finetuning
CLUSTER=ai2/allennlp-cirrascale

# generic gantry command for this.
while read line; do
      arr=($line)
      og_name=${arr[0]}
      dataset_id=${arr[1]}
      chkpt=$(beaker dataset ls ${dataset_id} | grep -oPe checkpoint\-[0-9]+ | tail -n 1)
      gantry run -y -n ${og_name}-evaluate --allow-dirty \
        --workspace $WORKSPACE \
        --gpus 1 \
        --nfs \
        --dataset ${dataset_id}:/inputs \
        --priority preemptible \
        --cluster ai2/allennlp-cirrascale \
        --cluster ai2/mosaic-cirrascale-a100 \
        --cluster ai2/general-cirrascale-a100-80g-ib \
        --cluster ai2/general-cirrascale \
        --env 'HF_HOME=/net/nfs.cirrascale/allennlp/hamishi/.hf' \
        --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
        --env 'HF_DATASETS_OFFLINE=1' \
        --env 'TRANSFORMERS_OFFLINE=1' \
        --beaker-image 'ai2/pytorch1.13.0-cuda11.6-python3.9' \
        --venv 'base' \
        --pip requirements.txt \
        --dry-run \
        --save-spec tmp.yaml \
        -- python minimal_multitask/evaluate_sni.py --model /inputs --output_folder /results
      sed -i "s/\- mountPath\: \/inputs/\- mountPath\: \/inputs\n    subPath\: ${chkpt}/g" tmp.yaml

      beaker experiment create tmp.yaml -n ${og_name}-evaluate-sni -p preemptible -w $WORKSPACE
done < data/experiment_ids.txt

rm tmp.yaml
