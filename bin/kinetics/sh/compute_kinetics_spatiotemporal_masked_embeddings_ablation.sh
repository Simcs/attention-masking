for spatial_threshold in 0.01
do
    for mode in outside
    do
        for dataset in k400 k600 k700-2020
        do
            echo python -m bin.compute_kinetics_spatiotemporal_masked_embeddings --dataset ${dataset} --split val \
                --spatial_masking --spatial_mask_method lt_threshold --spatial_mask_threshold ${spatial_threshold} \
                --temporal_masking --temporal_mask_ratio 0.5 --temporal_mode ${mode}
            python -m bin.compute_kinetics_spatiotemporal_masked_embeddings \
                --dataset ${dataset} \
                --split val \
                --spatial_masking \
                --spatial_mask_method lt_threshold \
                --spatial_mask_threshold ${spatial_threshold} \
                --temporal_masking \
                --temporal_mask_method random \
                --temporal_mask_ratio 0.5 \
                --temporal_mode ${mode}
        done
    done
done