#!/bin/bash

declare -a REMAP_TYPES=("freq-high" "freq-low" "random")
declare -a DATASETS=("sst2" "imdb")

for i in "${REMAP_TYPES[@]}"
do
    for j in "${DATASETS[@]}"
    do
        sbatch attacker.sh $i $j
    done
   # or do whatever with individual element of the array
done
# BENCHMARK="imdb"
# SAVEPATH="/home/reemh÷a/Privacy-Playground/roberta_imdb"
#REMAP_TYPE=freq-high 
#REMAP_TYPE=freq-low 
#REMAP_TYPE=random
#DATASET=imdb 

# sbatch attacker.sh 


