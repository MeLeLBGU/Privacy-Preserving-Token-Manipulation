#!/bin/bash

#SBATCH --partition main 
#SBATCH --time 6-10:30:00 
#SBATCH --job-name t5flan
#SBATCH --output t5flan.out
#SBATCH --gpus=1

python t5_flan_mmlu.py --finetune
