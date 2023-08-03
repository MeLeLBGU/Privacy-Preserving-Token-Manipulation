#!/bin/bash
#SBATCH --partition main ### partition name where to run a job. Use ‘main’ unless qos is required. qos partitions ‘rtx3090’ ‘rtx2080’ ‘gtx1080’
#SBATCH --time 6-23:30:00 ### limit the time of job running. Make sure it is not greater than the partition time limit (7 days)!! Format: D-H:MM:SS
#SBATCH --job-name privacy ### name of the job. replace my_job with your desired job name
#SBATCH --output slurm-%J.out ### output log for running job - %J is the job number variable
#SBATCH --gpus=1 ### number of GPUs. Choosing type e.g.: #SBATCH --gpus=gtx_1080:1 , or rtx_2080, or rtx_3090 . Allocating more than 1 requires the IT team’s permission
##SBATCH --tasks=1 # 1 process – use for processing of few programs concurrently in a job (with srun). Use just 1 otherwise

module load cuda/11.2
#module load anaconda
#source activate Privacy
#conda activate Privacy
nvidia-smi -L
#REMAP_TYPE=freq-high 
#REMAP_TYPE=freq-low 
#REMAP_TYPE=random
#DATASET=imdb 
#DATASET=sst2 
#FW=half

python main.py --frequency_path=/home/reemha/Privacy-Playground/wiki_freq_roberta.pkl --remap_type=$1 --frequency_window=half --dataset=$2 --attacker #--finetune

#python main.py --frequency_path=/home/reemha/Privacy-Playground/wiki_freq_roberta.pkl --remap_type=$1 --dataset=$2 --attacker --remap=all --frequency_window=half
#high
#python main.py --frequency_path=/home/reemha/Privacy-Playground/wiki_freq_roberta.pkl --remap_type=freq-high --dataset=sst2 --attacker --remap=all --frequency_window=half

#low
# python main.py --frequency_path=/home/reemha/Privacy-Playground/wiki_freq_roberta.pkl --remap_type=freq-low --dataset=imdb --attacker --remap=all --frequency_window=half

#random
# python main.py --remap_count=2 --frequency_path=/home/reemha/Privacy-Playground/wiki_freq_roberta.pkl --remap_type=random --dataset=sst2 --attacker --remap=all


