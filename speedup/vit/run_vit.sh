#!/bin/bash

max_seq_len=2048
num_head=12
head_dim=64
latent_size=$(( $num_head * $head_dim ))

python main.py \
    --patch-size 8 \
    --latent-size $latent_size \
    --num-heads $num_head \
    --num-encoders 12 \
    --img-size 384 \
    --num-classes 1024 \
    --batch-size 8 \
    --max_train_steps 8
