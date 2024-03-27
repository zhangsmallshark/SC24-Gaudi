#!/bin/bash --login
#PBS -l select=1:system=polaris
#PBS -l walltime=02:30:00
#PBS -l filesystems=home:eagle
#PBS -q preemptable
#PBS -A deepspeed_collab

# preemptable

export HF_HOME=/lus/eagle/projects/MDClimSim/chengming/huggingface
export HF_DATASETS_CACHE=/lus/eagle/projects/MDClimSim/chengming/huggingface/datasets
export TRANSFORMERS_CACHE=/lus/eagle/projects/MDClimSim/chengming/huggingface/models

export TORCH_EXTENSIONS_DIR=/home/czh5/.cache/polaris_torch_extensions

module load conda/2023-10-04 ; conda activate base
source /home/czh5/att-e/venvs/polaris/2023-10-04/bin/activate

echo "arguments: dataset_name method"
# model_name=$1
# if if (( $a == $b )) [ $dataset_name  = "wikitext" ]; then
#     echo 1
# fi

if [[ "$model" == "base" ]]; then
    echo $model
fi

model_name=gpt-neo-125m
# dataset_name=wikitext
# dataset_config_name=wikitext-103-raw-v1
dataset_name=bookcorpus
dataset_config_name=""
type=book_base0
output_dir=/lus/eagle/projects/MDClimSim/chengming/att-e/gpt-neo-125m/$type
# rm -rf $output_dir/*
rm -rf /home/czh5/att-e/gpt2/wandb/*

export WANDB_PROJECT=$model_name

WK_DIR=/home/czh5/att-e/gpt2
cd $WK_DIR

# --max_steps 5000 \
# --warmup_steps 1000 \
# --evaluation_strategy epoch \
# --dataset_config_name wikitext-2-raw-v1 \

# python run_clm.py \
python -m accelerate.commands.launch --num_processes=4 run_clm.py \
    --model_name_or_path EleutherAI/gpt-neo-125m \
    --dataset_name $dataset_name \
    --dataloader_num_workers 2 \
    --disable_tqdm True \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --report_to wandb \
    --run_name gpt-125m-$dataset_name-$type \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --logging_steps 50 \
    --learning_rate 0.0006 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --weight_decay 0.10 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 40 \
    --save_strategy steps \
    --save_steps 400 \
    --save_total_limit 6 \
    --output_dir $output_dir
