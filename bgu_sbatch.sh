#!/bin/bash
#SBATCH --partition main ### partition name where to run a job. Use ‘main’ unless qos is required. qos partitions ‘rtx3090’ ‘rtx2080’ ‘gtx1080’
#SBATCH --time 0-10:30:00 ### limit the time of job running. Make sure it is not greater than the partition time limit (7 days)!! Format: D-H:MM:SS
#SBATCH --job-name privacy ### name of the job. replace my_job with your desired job name
#SBATCH --output slurm-%J.out ### output log for running job - %J is the job number variable
#SBATCH --gpus=1 ### number of GPUs. Choosing type e.g.: #SBATCH --gpus=gtx_1080:1 , or rtx_2080, or rtx_3090 . Allocating more than 1 requires the IT team’s permission
##SBATCH --tasks=1 # 1 process – use for processing of few programs concurrently in a job (with srun). Use just 1 otherwise

module load cuda/11.2
#module load anaconda
#source activate Privacy
#conda activate Privacy
nvidia-smi -L
# for run all
#python main.py --frequency_path=/home/reemha/Privacy-Playground/wiki_freq_roberta.pkl --remap=$1 --save=$2 --remap_type=$3 --frequency_window=half --dataset=$4 --remap_count=$5 #--finetune

# python main.py --frequency_path=/home/reemha/Privacy-Playground/wiki_freq_roberta.pkl --remap=all --save=stencil_sst2_all.pt --remap_type=conv --frequency_window=half --dataset=sst2  #--finetune
# python main.py --remap_type=random --remap=all --model=roberta-base --dataset=imdb --save="imdb_512_random_all.pt"
#python main.py --frequency_path=/home/reemha/Privacy-Playground/wiki_freq_roberta.pkl --save=$1 --remap_type=$2 --frequency_window=$3 --dataset=$4 #--finetune
#python create_frequencies.py --save=wiki_freq_roberta2.pkl --model=roberta-base

