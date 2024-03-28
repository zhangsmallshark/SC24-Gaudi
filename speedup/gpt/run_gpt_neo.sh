#!/bin/bash

max_seq_len=2048

OUT_DIR=gpt_neo_outputs
# mkdir -p $OUT_DIR

# train
# python run_gpt_neo.py \
#     --num_train_epochs 1 \
#     --max_train_steps 7 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --n_embd_per_head 48 \
#     --n_layer 12 \
#     --n_head 16 \
#     --block_size ${max_seq_len} \
#     --n_positions ${max_seq_len} \
#     --output_dir $OUT_DIR

# test
python run_gpt_neo.py \
    --max_test_steps 10 \
    --per_device_eval_batch_size 4 \
    --n_embd_per_head 64 \
    --n_layer 4 \
    --n_head 12 \
    --n_positions ${max_seq_len} \
    --output_dir $OUT_DIR
