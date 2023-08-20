for spatial_threshold in 0.01
do
    for temporal_threhold in 0.05
    do
        for T in 1.5
        do
            echo python -m bin.compute_ucf_hmdb_masked_energy \
                --spatial_masking --spatial_mask_method lt_threshold --spatial_mask_threshold ${spatial_threshold} \
                --temporal_masking --temporal_mask_method lt_threshold --temporal_mask_threshold ${temporal_threhold} \
                --T ${T}
            python -m bin.compute_ucf_hmdb_masked_energy \
                --spatial_masking \
                --spatial_mask_method lt_threshold \
                --spatial_mask_threshold ${spatial_threshold} \
                --temporal_masking \
                --temporal_mask_method lt_threshold \
                --temporal_mask_threshold ${temporal_threhold} \
                --T ${T}
        done
    done
done