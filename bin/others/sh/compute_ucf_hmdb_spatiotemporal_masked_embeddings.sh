# for spatial_threshold in 0.005 0.01 0.05 0.1
# do
#     for temporal_threhold in 0.01 0.05 0.1
#     do
#         for fold in 3
#         do
#             for split in train
#             do
#                 echo python -m bin.compute_ucf_hmdb_spatiotemporal_masked_embeddings_2 --dataset ucf101 --fold ${fold} --split ${split} \
#                     --spatial_masking --spatial_mask_method lt_threshold --spatial_mask_threshold ${spatial_threshold} \
#                     --temporal_masking --temporal_mask_method lt_threshold --temporal_mask_threshold ${temporal_threhold} --temporal_mode default
#                 python -m bin.compute_ucf_hmdb_spatiotemporal_masked_embeddings_2 \
#                     --dataset ucf101 \
#                     --fold ${fold} \
#                     --split ${split} \
#                     --spatial_masking \
#                     --spatial_mask_method lt_threshold \
#                     --spatial_mask_threshold ${spatial_threshold} \
#                     --temporal_masking \
#                     --temporal_mask_method lt_threshold \
#                     --temporal_mask_threshold ${temporal_threhold} \
#                     --temporal_mode default \

#             done
#         done
#     done
# done

for spatial_threshold in 0.01
do
    for temporal_threhold in 0.05
    do
        for fold in 1 2 3
        do
            echo python -m bin.compute_ucf_hmdb_spatiotemporal_masked_embeddings --dataset hmdb51 --fold ${fold} --split test \
                --spatial_masking --spatial_mask_method lt_threshold --spatial_mask_threshold ${spatial_threshold} \
                --temporal_masking --temporal_mask_method lt_threshold --temporal_mask_threshold ${temporal_threhold}
            python -m bin.compute_ucf_hmdb_spatiotemporal_masked_embeddings \
                --dataset hmdb51 \
                --fold ${fold} \
                --split test \
                --spatial_masking \
                --spatial_mask_method lt_threshold \
                --spatial_mask_threshold ${spatial_threshold} \
                --temporal_masking \
                --temporal_mask_method lt_threshold \
                --temporal_mask_threshold ${temporal_threhold} \
                --temporal_mode default \

        done
    done
done