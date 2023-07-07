WORKSPACE=ai2/minimal-multitask-finetuning
CLUSTER=ai2/allennlp-cirrascale

while read line; do
    lineparts=($line)
    name=${lineparts[0]}
    dataset_id=${lineparts[1]}
    chkpt=$(beaker dataset ls ${dataset_id} | grep -oPe checkpoint\-[0-9]+ | tail -n 1)
    gantry run -y -n ${name} --allow-dirty \
        --workspace ai2/minimal-multitask-finetuning \
        --gpus 1 \
        --nfs \
        --task-name ${name}_mmlu_eval \
        --dataset ${dataset_id}:/inputs \
        --env 'HF_HOME=/net/nfs.cirrascale/allennlp/hamishi/.hf' \
        --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
        --env 'HF_DATASETS_OFFLINE=1' \
        --env 'TRANSFORMERS_OFFLINE=1' \
        --priority preemptible \
        --cluster  ai2/general-cirrascale \
        --cluster ai2/allennlp-cirrascale \
        --cluster ai2/general-cirrascale-a100-80g-ib \
        --beaker-image 'ai2/pytorch1.13.0-cuda11.6-python3.9' \
        --venv 'base' \
        --dry-run \
        --allow-dirty \
        --save-spec tmp.yaml \
        --pip requirements.txt \
        -- python -u scripts/mmlu/eval_mmlu.py --model /inputs --ntrain 5 --ngpu 1 --data_dir /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/mmlu --save_dir /results
    sed -i "s/\- mountPath\: \/inputs/\- mountPath\: \/inputs\n    subPath\: ${chkpt}/g" tmp.yaml
    beaker experiment create tmp.yaml -n ${name}_mmlu_eval_5shot --priority preemptible -w $WORKSPACE
done < beaker_scripts/datasets_to_load.txt
rm tmp.yaml