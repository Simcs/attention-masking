python -m finetune_vivit \
    -seed 1 \
    -epoch 10 \
    -batch_size 12 \
    -num_workers 6 \
    -log_interval 1 \
    -gpus 1 \
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
    -pretrain_pth ./logs/vivit/vivit_model.pth \
    -weights_from kinetics \
    -attention_type fact_encoder \
    -dataset ucf101 \
    -use_fp16 False \
    # -resume \
    # -resume_from_checkpoint True \