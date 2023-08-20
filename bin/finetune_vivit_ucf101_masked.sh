python -m finetune_vivit \
    -epoch 20 \
    -batch_size 12 \
    -num_workers 6 \
    -log_interval 1 \
    -gpus 0 1 \
    -root_dir ./logs \
    -objective supervised \
    -num_class 101 \
    -fold 1 \
    -img_size 224 \
    -arch vivit \
    -train_data_path ~/workspace/dataset/ucf101 \
    -val_data_path ~/workspace/dataset/ucf101 \
    -lr 0.0001 \
    -num_frames 16 \
    -frame_interval 1 \
    -weight_decay 0.0001 \
    -warmup_epochs 1 \
    -pretrain_pth "/home/simc/workspace/OOD-with-ViT/logs/results/vivit_pretrained_ucf101_fp32_fold_1_aug/ckpt/2023-03-13 22:38:24_ep_0_top1_acc_0.895.pth" \
    -resume_from_checkpoint "/home/simc/workspace/OOD-with-ViT/logs/results/vivit_pretrained_ucf101_fp32_fold_1_aug/ckpt/2023-03-13 22:38:24_ep_0_top1_acc_0.895.pth" \
    -weights_from kinetics \
    -attention_type fact_encoder \
    -dataset ucf101 \
    -use_fp16 False \
    -resume \
    # -attention_masking \
    # -resume_from_checkpoint True \