set -ex
models=(tulu_squad_context_10ep tulu_gsm8k_shots_10ep tulu_codex_10ep tulu_tydiqa_shots_10ep tulu_bbh_shots_10ep tulu_mmlu_shots_10ep tulu_alpacafarm_10ep)
beaker_paths=(01HTZSBFNK2SJQ58RZJN1Q146N 01HTZSBFG2J8WR8DB6QGV9DRRC 01HTZSBFAHVH4P21YV79S8PMBQ 01HTZSB5KK618KVKDW3949593G 01HTZSB5DBT22MRDKV0ZP98JQG 01HTZSB56GFDVV7764Y49RQ8ST 01HTZSB50DSK0QQFP3Q4MMRRYX)

for i in {0..6}
do
    model=${models[i]}
    beaker_path=${beaker_paths[i]}
    for subpath in checkpoint-78 checkpoint-156 checkpoint-234 checkpoint-312 checkpoint-390 checkpoint-468 checkpoint-546 checkpoint-625 checkpoint-703 checkpoint-780
    do
        ./shell_scripts/gantry_commands/eval_beaker.sh ${model}_${subpath} $beaker_path $subpath
    done
done

./shell_scripts/gantry_commands/eval_beaker.sh tydiqa_train_10k 01HVHC8K3T5B3MX3VN9VK6VR7W
./shell_scripts/gantry_commands/eval_beaker.sh squad_train_10k 01HVHC96FJ5S61CQV7BPREBPBQ

./shell_scripts/gantry_commands/eval_beaker.sh tydiqa_train 01HVHC8K3T5B3MX3VN9VK6VR7W
./shell_scripts/gantry_commands/eval_beaker.sh squad_train 01HVFKCGRA1HRJGK0KT62Y78RC
./shell_scripts/gantry_commands/eval_beaker.sh squad_tulu_combined_sel 01HVKXT1459XN0YRDF6EWF0XMK


./shell_scripts/gantry_commands/eval_beaker.sh fixed_domain_afarm_unfiltered 01HX0BHBNF6MAEZF4Z4RY6JNJW