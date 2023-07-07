
WORKSPACE=ai2/minimal-multitask-finetuning
CLUSTER=ai2/allennlp-cirrascale

# generic gantry command for this.
while read line; do
      arr=($line)
      og_name=${arr[0]}
      dataset_id=${arr[1]}
      beaker group add 01GRZ4WMAM3D9KDB3CMT1GMV8F ${dataset_id}
done < data/experiment_ids.txt

rm tmp.yaml
