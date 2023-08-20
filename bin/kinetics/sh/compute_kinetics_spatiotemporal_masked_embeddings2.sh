for spatial_threshold in 0.01
do
    for temporal_threhold in 0.7 0.8 0.9
    do
        for dataset in k400 k600 k700-2020
        do
            echo python -m bin.compute_kinetics_spatiotemporal_masked_embeddings --dataset ${dataset} --split val \
                --spatial_masking --spatial_mask_method lt_threshold --spatial_mask_threshold ${spatial_threshold} \
                --temporal_masking --temporal_mask_method lt_threshold --temporal_mask_threshold ${temporal_threhold}
            python -m bin.compute_kinetics_spatiotemporal_masked_embeddings \
                --dataset ${dataset} \
                --split val \
                --spatial_masking \
                --spatial_mask_method lt_threshold \
                --spatial_mask_threshold ${spatial_threshold} \
                --temporal_masking \
                --temporal_mask_method lt_threshold \
                --temporal_mask_threshold ${temporal_threhold}
        done
    done
done