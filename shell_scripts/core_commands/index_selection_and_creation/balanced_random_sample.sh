for datasize in 2500000; do
    for seed in 42; do
        python scripts/construct_downsampled_balanced.py \
            --input_file /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/tulu_v2_unfiltered_data_dedup.jsonl \
            --output_file /net/nfs.cirrascale/allennlp/muruz/influence-tuning/selections/tulu2_unfiltered_subsample/balance_random_${datasize}_seed${seed}.json \
            --seed $seed \
            --num_samples $datasize
    done
done


for datasize in 10000 25000 50000 67000 100000 180000 326000; do
    for seed in 42 0 1; do
        python scripts/sample_tulu2.py \
            --ori_dataset /net/nfs.cirrascale/allennlp/hamishi/minimal-multitask-tuning/data/tulu_splits/tulu_v2_unfiltered_fixed/tulu_v2_unfiltered/tulu_v2_unfiltered_data_dedup.jsonl \
            --distribution_file auxiliary_data/tulu2_unfiltered_distrib.json \
            --subset_size $datasize \
            --output_file /net/nfs.cirrascale/allennlp/muruz/influence-tuning/selections/tulu2_unfiltered_subsample/distmatching_random_${datasize}_seed${seed}.json \
            --seed $seed
    done
done