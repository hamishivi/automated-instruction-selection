#!/bin/bash

BASEDIR=$(pwd)
BASEDIR=$(readlink -f ${BASEDIR})
OUTPUTDIR=${BASEDIR}/sbatch_output

mkdir -p jobs

BATCH_FILE=$(mktemp -p jobs --suffix=.job)

cat > ${BATCH_FILE} <<EOF
#!/bin/bash -x
#SBATCH --job-name=eval_mbpp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0-24:00:00
#SBATCH --gpus=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --partition=${2}
#SBATCH --account=${1}
#SBATCH --output=${OUTPUTDIR}/eval_mbpp_${3}.out
export HF_ALLOW_CODE_EVAL="1"
python3 -m minimal_multitask.eval.mbpp.run_eval --model_name_or_path results/llama_2_128lora_tulu2/${3}/ --save_dir results/llama_2_128lora_tulu2/${3}/ --eval_batch_size 32 --eval_pass_at_ks 10 --temperature 0.8 --use_chat_format --unbiased_sampling_size_n 10
exit 0

EOF

sbatch ${BATCH_FILE}