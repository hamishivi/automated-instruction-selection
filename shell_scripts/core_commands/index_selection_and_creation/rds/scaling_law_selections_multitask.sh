for datasize in 1200000 2500000; do
        python -m minimal_multitask.get_top_aggregated_influences \
                --input_files rds_pickles/*_cossim.json \
                --output_file selections/rds_tulu2_unfiltered/rds_multitask_rrmax${datasize}.json \
                --output_size $datasize --selection_method max \
                --train_dataset ../../hamishi/influence-tuning/data/tulu_v2_unfiltered_data_dedup.jsonl \
                --maximize_inf \
                --output_dataset
done