#!/bin/bash --login
#PBS -l select=1:system=polaris
#PBS -l walltime=2:20:00
#PBS -l filesystems=home:eagle
#PBS -q preemptable
#PBS -A MDClimSim

# huggingface-cli login
# export HF_HOME=~/.huggingface
export HF_HOME=/lus/eagle/projects/MDClimSim/chengming/huggingface
export HF_DATASETS_CACHE=/lus/eagle/projects/MDClimSim/chengming/huggingface/datasets
export TRANSFORMERS_CACHE=/lus/eagle/projects/MDClimSim/chengming/huggingface/models

export TORCH_EXTENSIONS_DIR=/home/czh5/.cache/polaris_torch_extensions

# module load conda/2023-10-04 ; conda activate base
# source /home/czh5/att-e/venvs/polaris/2023-10-04/bin/activate

echo "arguments format: model"
model=$1
model=baseline
output_dir=/lus/eagle/projects/MDClimSim/chengming/att-e/vit-imagenet-1k/$model
# rm -rf $output_dir/*
rm -rf /home/czh5/att-e/vit/wandb/*

export WANDB_PROJECT=vit-$model

# --max_steps 5000 \
# --warmup_ratio 0.1 \
# --warmup_steps 1000 \
# --evaluation_strategy epoch \
# --weight_decay 0.10 \

# python run_image_classification.py \
python -m accelerate.commands.launch --num_processes=4 run_image_classification.py \
    --model_name_or_path google/vit-base-patch16-384 \
    --dataset_name imagenet-1k \
    --remove_unused_columns False \
    --dataloader_num_workers 2 \
    --disable_tqdm True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --do_train \
    --do_eval \
    --report_to wandb \
    --run_name vit-imagenet-1k-$model \
    --evaluation_strategy steps \
    --eval_steps 20000 \
    --logging_steps 50 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type linear \
    --gradient_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps 2000 \
    --num_train_epochs 3 \
    --save_total_limit 6 \
    --output_dir $output_dir

