for spatial_threshold in 0.005 0.01 0.05 0.1
do
    for temporal_threhold in 0.01 0.05 0.1
    do
        for fold in 1 2 3
        do
            echo python -m bin.compute_ucf_masked_statistics --fold ${fold} \
                --spatial_masking --spatial_mask_method lt_threshold --spatial_mask_threshold ${spatial_threshold} \
                --temporal_masking --temporal_mask_method lt_threshold --temporal_mask_threshold ${temporal_threhold} --temporal_mode default
            python -m bin.compute_ucf_masked_statistics \
                --fold ${fold} \
                --spatial_masking \
                --spatial_mask_method lt_threshold \
                --spatial_mask_threshold ${spatial_threshold} \
                --temporal_masking \
                --temporal_mask_method lt_threshold \
                --temporal_mask_threshold ${temporal_threhold}
        done
    done
done