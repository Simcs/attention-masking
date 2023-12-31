for spatial_threshold in 0.01
do
    for temporal_threhold in 0.05
    do
        for shot in 50
        do
            echo python -m bin.compute_kinetics_masked_few_shot_oe \
                --spatial_masking --spatial_mask_method lt_threshold --spatial_mask_threshold ${spatial_threshold} \
                --temporal_masking --temporal_mask_method lt_threshold --temporal_mask_threshold ${temporal_threhold} \
                --n_shot ${shot}
            python -m bin.compute_kinetics_masked_few_shot_oe \
                --spatial_masking \
                --spatial_mask_method lt_threshold \
                --spatial_mask_threshold ${spatial_threshold} \
                --temporal_masking \
                --temporal_mask_method lt_threshold \
                --temporal_mask_threshold ${temporal_threhold} \
                --n_shot ${shot}
        done
    done
done