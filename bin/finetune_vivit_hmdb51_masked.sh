python -m finetune_vivit \
    -epoch 10 \
    -batch_size 16 \
    -num_workers 16 \
    -log_interval 1 \
    -gpus 0 1 \
    -root_dir ./logs \
    -objective supervised \
    -num_class 101 \
    -img_size 224 \
    -arch vivit \
    -train_data_path ~/workspace/dataset/hmdb51 \
    -lr 0.001 \
    -num_frames 16 \
    -frame_interval 1 \
    -weight_decay 0.0001 \
    -warmup_epochs 1 \
    -pretrain_pth ./logs/vivit/vivit_model.pth \
    -weights_from kinetics \
    -attention_type fact_encoder \
    -dataset hmdb51 \