for ratio in 0.05 0.1 0.15 0.2
do
    for dataset in k400
    do
        echo python -m bin.compute_kinetics_masked_embeddings --dataset ${dataset} --split train \
            --mask_mode 'zero' --spatial_masking --spatial_mask_method lt_threshold --spatial_mask_threshold ${threshold}
        python -m bin.compute_kinetics_masked_embeddings \
            --dataset ${dataset} \
            --mask_mode 'zero' \
            --split train \
            --spatial_masking \
            --spatial_mask_method bottom_ratio \
            --spatial_mask_ratio ${ratio}
    done
done