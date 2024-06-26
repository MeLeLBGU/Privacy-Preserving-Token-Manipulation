#!/bin/bash

#SBATCH --partition main 
#SBATCH --time 6-10:30:00 
#SBATCH --job-name qnli
#SBATCH --output runtest.out
#SBATCH --gpus=1
#SBATCH --mem=0
module load cuda/11.2
#python reemtest.py
python main.py --model=t5-base --save=t5basetest --remap=all --stencil_stride=1 --stencil_sigma=0.6 --remap_type=conv --dataset=qnli --save=finetune-t5-base-qnli --stencil_size=11 --finetune
