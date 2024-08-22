set -ex

while IFS= read -r line; do
  id=$(echo $line | awk '{print $1}')
  name=$(echo $line | awk '{print $2}')
  ./shell_scripts/gantry_commands/compute_losses_beaker.sh $name $id
done < /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/shell_scripts/gantry_commands/beaker_ids_tulu2_select.txt